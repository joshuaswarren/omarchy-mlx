# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""The serve status line for the vendored mlx-lm patches.

Changed files alone do not prove a complete apply, so "applied" needs the
completion record the apply script writes after its last step. A synthetic
mlx-lm distribution (no network, no real mlx-lm) covers the four states plus
bounded non-blocking reads across METADATA/PKG-INFO, RECORD, mlx_lm/*.py, and
mlx_lm/.omarchy-serve-patches.
"""

import base64
import contextlib
import hashlib
import importlib.metadata
import os
import pathlib
import signal
import socket
import stat
import sys
import tempfile
import types
from unittest import mock

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "serve"))

from mlx_omarchy_serve import __main__ as serve_main  # noqa: E402
from mlx_omarchy_serve import _mlxlm_server as cap_shim  # noqa: E402

VERSION = "9.9.9"
RECORD_NAME = ".omarchy-serve-patches"
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


def _truncate_sparse(path: pathlib.Path, size: int) -> None:
    try:
        with open(path, "wb") as fh:
            fh.truncate(size)
    except OSError as exc:
        pytest.skip(f"filesystem does not support sparse truncate: {exc}")


def _zero_size_stat(st: os.stat_result) -> os.stat_result:
    return os.stat_result((
        st.st_mode, st.st_ino, st.st_dev, st.st_nlink,
        st.st_uid, st.st_gid, 0, st.st_atime, st.st_mtime,
        st.st_ctime,
    ))


def _b64_sha256(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _digest(data: bytes) -> str:
    return "sha256=" + _b64_sha256(data)


def _dist_info(sp: pathlib.Path) -> pathlib.Path:
    return sp / f"mlx_lm-{VERSION}.dist-info"


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A site-packages with a fake mlx-lm 9.9.9 whose RECORD matches its files."""
    sp = tmp_path / "site"
    pkg = sp / "mlx_lm"
    pkg.mkdir(parents=True)
    files = {"mlx_lm/a.py": b"A = 1\n", "mlx_lm/b.py": b"B = 2\n"}
    lines = []
    for name, data in files.items():
        (sp / name).write_bytes(data)
        lines.append(f"{name},{_digest(data)},{len(data)}")
    info = sp / f"mlx_lm-{VERSION}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: mlx-lm\nVersion: {VERSION}\n")
    (info / "RECORD").write_text("\n".join(lines + [f"mlx_lm-{VERSION}.dist-info/METADATA,,", f"mlx_lm-{VERSION}.dist-info/RECORD,,"]) + "\n")
    monkeypatch.syspath_prepend(str(sp))
    return sp


def test_stock_tree_says_not_applied(site):
    assert "NOT applied" in serve_main.mlxlm_patch_status()


def test_changed_file_without_a_record_is_partial(site):
    (site / "mlx_lm/a.py").write_text("A = 1  # patched\n")
    line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "1 of 2" in line
    assert "applied (" not in line


def test_changed_file_with_a_matching_record_is_applied(site):
    (site / "mlx_lm/a.py").write_text("A = 1  # patched\n")
    (site / "mlx_lm" / RECORD_NAME).write_text(f"series=patches\nmlx_lm={VERSION}\n")
    line = serve_main.mlxlm_patch_status()
    assert line.startswith("mlx-lm serve patches: applied (1 of 2")
    assert "apply completed" in line


def test_a_record_for_another_version_does_not_count(site):
    (site / "mlx_lm/a.py").write_text("A = 1  # patched\n")
    (site / "mlx_lm" / RECORD_NAME).write_text("series=patches\nmlx_lm=0.0.1\n")
    assert "PARTIAL or unverified" in serve_main.mlxlm_patch_status()


def test_a_record_over_a_restored_tree_says_not_applied(site):
    (site / "mlx_lm" / RECORD_NAME).write_text(f"series=patches\nmlx_lm={VERSION}\n")
    assert "NOT applied" in serve_main.mlxlm_patch_status()


# ----------------------------------------------------------- absent & fake dist


def test_absent_mlxlm_returns_none(monkeypatch):
    def _missing(_name):
        raise importlib.metadata.PackageNotFoundError("mlx-lm")

    monkeypatch.setattr(importlib.metadata, "distribution", _missing)
    assert serve_main.mlxlm_patch_status() is None


def test_distribution_lookup_error_fails_closed(monkeypatch):
    def _broken(_name):
        raise OSError("metadata probe failed")

    monkeypatch.setattr(importlib.metadata, "distribution", _broken)
    line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "applied (" not in line
    assert "NOT applied" not in line


def test_missing_record_and_egg_info_fail_closed_while_editable_record_returns_none(tmp_path, monkeypatch):
    sp = tmp_path / "site"
    pkg = sp / "mlx_lm"
    pkg.mkdir(parents=True)
    (pkg / "a.py").write_bytes(b"A = 1\n")
    info = sp / f"mlx_lm-{VERSION}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: mlx-lm\nVersion: {VERSION}\n")
    monkeypatch.syspath_prepend(str(sp))
    # Missing RECORD in an installed .dist-info must fail closed (not appear absent).
    line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "RECORD" in line

    # Valid PEP 376 RECORD with only metadata / .pth rows (editable install) -> None.
    (info / "RECORD").write_text(
        f"mlx_lm-{VERSION}.dist-info/METADATA,,\n"
        f"__editable__.mlx_lm-{VERSION}.pth,{_digest(b'/src')},4\n"
    )
    assert serve_main.mlxlm_patch_status() is None

    # Single-file .egg-info has no wheel RECORD -> fails closed.
    egg = tmp_path / f"mlx_lm-{VERSION}.egg-info"
    egg.write_text(f"Metadata-Version: 2.1\nName: mlx-lm\nVersion: {VERSION}\n")
    egg_dist = importlib.metadata.PathDistribution(egg)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _name: egg_dist)
    line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "RECORD" in line

    # PathDistribution with a bytes-returning os.PathLike must not raise TypeError.
    class _BytesPathLike:
        def __fspath__(self):
            return os.fsencode(info)

    bytes_dist = importlib.metadata.PathDistribution(_BytesPathLike())
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _name: bytes_dist)
    line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert serve_main._read_mlxlm_dist_version(bytes_dist) is None


def test_monkeypatched_fake_dist_works_and_bounds_reads(tmp_path, monkeypatch):
    root = tmp_path / "fake_site"
    pkg = root / "mlx_lm"
    pkg.mkdir(parents=True)
    a_bytes = b"A = 1\n"
    (pkg / "a.py").write_bytes(a_bytes)

    class _FakePath(str):
        def __new__(cls, rel, hash_obj):
            obj = super().__new__(cls, rel)
            obj.hash = hash_obj
            return obj

    class _FakeDist:
        def __init__(self, files, version=VERSION, *, raise_version=False, raise_locate=False):
            self.files = files
            self._version = version
            self._raise_version = raise_version
            self._raise_locate = raise_locate

        @property
        def version(self):
            if self._raise_version:
                raise RuntimeError("version boom")
            return self._version

        def locate_file(self, path):
            if self._raise_locate:
                raise RuntimeError("locate boom")
            return root / str(path)

    entry = _FakePath("mlx_lm/a.py", types.SimpleNamespace(mode="sha256", value=_b64_sha256(a_bytes)))
    fake = _FakeDist([entry])
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _name: fake)
    assert "NOT applied" in serve_main.mlxlm_patch_status()

    # Patched + valid completion record -> applied.
    (pkg / "a.py").write_bytes(b"A = 2\n")
    (pkg / RECORD_NAME).write_text(f"series=patches\nmlx_lm={VERSION}\n")
    line = serve_main.mlxlm_patch_status()
    assert line.startswith("mlx-lm serve patches: applied (1 of 1")

    # Fake dist whose version property raises or returns a non-string fails closed.
    for bad_dist in (
        _FakeDist([entry], raise_version=True),
        _FakeDist([entry], version=123),
        _FakeDist([entry], version=None),
    ):
        monkeypatch.setattr(importlib.metadata, "distribution", lambda _name, d=bad_dist: d)
        line = serve_main.mlxlm_patch_status()
        assert "PARTIAL or unverified" in line
        assert "version could not be read" in line
        assert "applied (" not in line

    # Fake dist whose locate_file raises fails closed.
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _name: _FakeDist([entry], raise_locate=True))
    line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "applied (" not in line

    # Invalid UTF-8 in patch record via fake dist -> PARTIAL or unverified (no crash).
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _name: fake)
    (pkg / RECORD_NAME).write_bytes(b"\xff\xfe\n")
    line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "applied (" not in line

    # FIFO at .py via fake dist must not block and must fail closed.
    (pkg / "a.py").unlink()
    os.mkfifo(pkg / "a.py")
    with _watchdog(5):
        line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "NOT applied" not in line
    assert "applied (" not in line

    # Lazy generator raising mid-iteration in dist.files fails closed cleanly.
    def _boom_gen():
        yield entry
        raise RuntimeError("boom")

    monkeypatch.setattr(importlib.metadata, "distribution", lambda _name: _FakeDist(_boom_gen()))
    line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line

    # Fake dist with files=None or files=[] fails closed; non-py files list returns None.
    for empty_files in (None, []):
        monkeypatch.setattr(importlib.metadata, "distribution", lambda _name, ef=empty_files: _FakeDist(ef))
        assert "PARTIAL or unverified" in serve_main.mlxlm_patch_status()
    pth = _FakePath("__editable__.mlx_lm-9.9.9.pth", None)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _name: _FakeDist([pth]))
    assert serve_main.mlxlm_patch_status() is None


# ---------------------------------------------------- bounded mlx_lm/*.py reads


def test_empty_regular_py_verifies_while_dev_null_fails_closed(site):
    info = _dist_info(site)
    (site / "mlx_lm/a.py").write_bytes(b"")
    b_data = (site / "mlx_lm/b.py").read_bytes()
    (info / "RECORD").write_text(
        f"mlx_lm/a.py,{_digest(b'')},0\n"
        f"mlx_lm/b.py,{_digest(b_data)},{len(b_data)}\n"
    )
    assert "NOT applied" in serve_main.mlxlm_patch_status()

    # Replacing a.py with a symlink to /dev/null must fail closed without reading.
    (site / "mlx_lm/a.py").unlink()
    (site / "mlx_lm/a.py").symlink_to("/dev/null")
    real_read = os.read
    read_non_reg = []

    def _guard_read(fd, n):
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            read_non_reg.append(n)
        return real_read(fd, n)

    with mock.patch.object(cap_shim.os, "read", side_effect=_guard_read):
        line = serve_main.mlxlm_patch_status()
    assert read_non_reg == []
    assert "PARTIAL or unverified" in line
    assert "1 of 2" in line
    assert "NOT applied" not in line


def test_py_symlink_to_regular_file_is_followed(site, tmp_path):
    target = tmp_path / "real_a.py"
    target.write_bytes(b"A = 1\n")
    link = site / "mlx_lm/a.py"
    link.unlink()
    link.symlink_to(target)
    assert "NOT applied" in serve_main.mlxlm_patch_status()

    target.write_bytes(b"A = 1  # patched\n")
    (site / "mlx_lm" / RECORD_NAME).write_text(f"series=patches\nmlx_lm={VERSION}\n")
    line = serve_main.mlxlm_patch_status()
    assert line.startswith("mlx-lm serve patches: applied (1 of 2")


def test_py_fifo_without_and_with_writer_fails_closed_without_blocking(site):
    target = site / "mlx_lm/a.py"
    target.unlink()
    os.mkfifo(target)
    with _watchdog(5):
        line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "1 of 2" in line
    assert "NOT applied" not in line

    # Writer present with stock bytes queued: still refused without consuming bytes.
    peer = os.open(target, os.O_RDWR | os.O_NONBLOCK)
    try:
        os.write(peer, b"A = 1\n")
        with _watchdog(5):
            line = serve_main.mlxlm_patch_status()
        assert "PARTIAL or unverified" in line
        assert os.read(peer, 64) == b"A = 1\n"
    finally:
        os.close(peer)


def test_py_sparse_4gib_fails_closed_without_reading(site):
    target = site / "mlx_lm/a.py"
    _truncate_sparse(target, 4 << 30)
    real_read = os.read
    read_sparse = []

    def _guard_read(fd, n):
        if os.fstat(fd).st_size == (4 << 30):
            read_sparse.append(n)
        return real_read(fd, n)

    with mock.patch.object(cap_shim.os, "read", side_effect=_guard_read):
        line = serve_main.mlxlm_patch_status()
    assert read_sparse == []
    assert "PARTIAL or unverified" in line
    assert "1 of 2" in line
    assert "NOT applied" not in line


def test_py_exact_cap_succeeds_and_cap_plus_one_fails_closed(site):
    cap = serve_main.MLXLM_PY_MAX_BYTES
    info = _dist_info(site)
    b_data = (site / "mlx_lm/b.py").read_bytes()

    at_cap = b"#" * cap
    (site / "mlx_lm/a.py").write_bytes(at_cap)
    (info / "RECORD").write_text(
        f"mlx_lm/a.py,{_digest(at_cap)},{len(at_cap)}\n"
        f"mlx_lm/b.py,{_digest(b_data)},{len(b_data)}\n"
    )
    assert "NOT applied" in serve_main.mlxlm_patch_status()

    over_cap = b"#" * (cap + 1)
    (site / "mlx_lm/a.py").write_bytes(over_cap)
    (info / "RECORD").write_text(
        f"mlx_lm/a.py,{_digest(over_cap)},{len(over_cap)}\n"
        f"mlx_lm/b.py,{_digest(b_data)},{len(b_data)}\n"
    )
    line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "1 of 2" in line
    assert "NOT applied" not in line


def test_py_zero_st_size_within_cap_and_lying_fstat_stops_at_cap_plus_one(site):
    target = site / "mlx_lm/a.py"
    target_ino = target.stat().st_ino
    real_fstat = os.fstat

    def _fake_fstat(fd):
        st = real_fstat(fd)
        return _zero_size_stat(st) if st.st_ino == target_ino else st

    with mock.patch.object(cap_shim.os, "fstat", side_effect=_fake_fstat):
        assert "NOT applied" in serve_main.mlxlm_patch_status()

    cap = serve_main.MLXLM_PY_MAX_BYTES
    data = b"#" * (cap + 4096)
    target.write_bytes(data)
    target_ino = target.stat().st_ino
    real_read = os.read
    seen = []

    def _counting_read(fd, n):
        chunk = real_read(fd, n)
        if real_fstat(fd).st_ino == target_ino:
            seen.append(len(chunk))
        return chunk

    with mock.patch.object(cap_shim.os, "fstat", side_effect=_fake_fstat), \
            mock.patch.object(cap_shim.os, "read", side_effect=_counting_read):
        line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert sum(seen) == cap + 1


def test_py_short_reads_reassemble_correctly(site):
    real_read = os.read
    calls = []

    def _short_read(fd, n):
        chunk = real_read(fd, min(n, 3))
        calls.append(len(chunk))
        return chunk

    with mock.patch.object(cap_shim.os, "read", side_effect=_short_read):
        assert "NOT applied" in serve_main.mlxlm_patch_status()
    assert len(calls) > 4


def test_py_unreadable_missing_dir_socket_symlink_loop_dangling_fail_closed(site):
    target = site / "mlx_lm/a.py"

    def _assert_unverified_both_modes():
        # Other file stock -> must NOT claim stock.
        (site / "mlx_lm/b.py").write_bytes(b"B = 2\n")
        patch_rec = site / "mlx_lm" / RECORD_NAME
        if patch_rec.exists():
            patch_rec.unlink()
        line_stock = serve_main.mlxlm_patch_status()
        assert "PARTIAL or unverified" in line_stock
        assert "1 of 2" in line_stock
        assert "NOT applied" not in line_stock
        assert "applied (" not in line_stock

        # Other file patched + valid completion record -> must NOT claim applied.
        (site / "mlx_lm/b.py").write_bytes(b"B = 2  # patched\n")
        patch_rec.write_text(f"series=patches\nmlx_lm={VERSION}\n")
        line_patched = serve_main.mlxlm_patch_status()
        assert "PARTIAL or unverified" in line_patched
        assert "1 of 2" in line_patched
        assert "applied (" not in line_patched

    # Missing file
    target.unlink()
    _assert_unverified_both_modes()

    # Directory named a.py
    target.mkdir()
    _assert_unverified_both_modes()
    target.rmdir()

    # Dangling symlink
    target.symlink_to(site / "missing.py")
    _assert_unverified_both_modes()
    target.unlink()

    # Symlink loop
    other = site / "loop_target"
    target.symlink_to(other)
    other.symlink_to(target)
    _assert_unverified_both_modes()
    target.unlink()
    other.unlink()

    # Unix socket
    with tempfile.TemporaryDirectory(dir="/tmp") as sd:
        sock_path = pathlib.Path(sd) / "s.sock"
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(str(sock_path))
            target.symlink_to(sock_path)
            _assert_unverified_both_modes()
        finally:
            target.unlink()
            sock.close()

    # Unreadable file (chmod 000) when non-root
    if os.geteuid() != 0:
        target.write_bytes(b"A = 1\n")
        target.chmod(0)
        try:
            _assert_unverified_both_modes()
        finally:
            target.chmod(0o600)


def test_open_flags_and_fd_cleanup_on_refusal_and_read_error(site):
    target = site / "mlx_lm/a.py"
    target.unlink()
    os.mkfifo(target)
    opened = []
    closed = []
    real_open = os.open
    real_close = os.close

    def _track_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        opened.append((pathlib.Path(path).name, fd, flags))
        return fd

    def _track_close(fd):
        closed.append(fd)
        return real_close(fd)

    with _watchdog(5), \
            mock.patch.object(cap_shim.os, "open", side_effect=_track_open), \
            mock.patch.object(cap_shim.os, "close", side_effect=_track_close):
        line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert opened
    for _name, fd, flags in opened:
        assert flags & os.O_NONBLOCK
        assert flags & os.O_CLOEXEC
        assert flags & os.O_NOCTTY
        assert fd in closed

    # Post-fstat read error on a.py must also close its fd cleanly.
    target.unlink()
    target.write_bytes(b"A = 1\n")
    a_ino = target.stat().st_ino
    opened.clear()
    closed.clear()
    real_read = os.read

    def _fail_a_read(fd, n):
        if os.fstat(fd).st_ino == a_ino:
            raise OSError("io fail")
        return real_read(fd, n)

    with mock.patch.object(cap_shim.os, "open", side_effect=_track_open), \
            mock.patch.object(cap_shim.os, "close", side_effect=_track_close), \
            mock.patch.object(cap_shim.os, "read", side_effect=_fail_a_read):
        line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert [fd for _n, fd, _f in opened] == closed


# ----------------------------------------- hash mode & canonical digest checks


def test_non_sha256_and_malformed_hashes_fail_closed_before_open(site):
    info = _dist_info(site)
    a_b64 = _b64_sha256(b"A = 1\n")
    b_b64 = _b64_sha256(b"B = 2\n")
    non_canonical = (
        a_b64[:-1]
        + _B64URL_ALPHABET[_B64URL_ALPHABET.index(a_b64[-1]) ^ 1]
    )
    bad_specs = [
        "md5=deadbeef",
        "sha1=deadbeef",
        f"sha512={a_b64}",
        "",
        "sha256",
        "sha256=",
        "sha256=!!",
        f"sha256=!!{a_b64}!!",
        f"sha256={a_b64}=",
        f"sha256={a_b64}==",
        f"sha256={a_b64[:-1]}+",
        f"sha256={a_b64[:-1]}/",
        f"sha256={non_canonical}",
    ]
    # Even with b.py modified and a matching completion record, a bad hash on
    # a.py must fail closed AND must not open a.py.
    (site / "mlx_lm/b.py").write_bytes(b"B = 2  # patched\n")
    (site / "mlx_lm" / RECORD_NAME).write_text(f"series=patches\nmlx_lm={VERSION}\n")
    real_open = os.open

    def _guard_open(path, *args, **kwargs):
        if pathlib.Path(path).name == "a.py":
            raise AssertionError("must not open a.py when its RECORD hash is invalid")
        return real_open(path, *args, **kwargs)

    for bad_spec in bad_specs:
        (info / "RECORD").write_text(
            f"mlx_lm/a.py,{bad_spec},6\n"
            f"mlx_lm/b.py,sha256={b_b64},6\n"
        )
        with mock.patch.object(cap_shim.os, "open", side_effect=_guard_open):
            line = serve_main.mlxlm_patch_status()
        assert "PARTIAL or unverified" in line, bad_spec
        assert "applied (" not in line, bad_spec
        assert "NOT applied" not in line, bad_spec


# --------------------------------------------------------- bounded RECORD reads


def test_record_fifo_without_and_with_writer_fails_closed(site):
    record = _dist_info(site) / "RECORD"
    record.unlink()
    os.mkfifo(record)
    with _watchdog(5):
        line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "RECORD" in line

    peer = os.open(record, os.O_RDWR | os.O_NONBLOCK)
    try:
        a_digest = _digest(b"A = 1\n")
        payload = f"mlx_lm/a.py,{a_digest},6\n".encode()
        os.write(peer, payload)
        with _watchdog(5):
            line = serve_main.mlxlm_patch_status()
        assert "PARTIAL or unverified" in line
        assert os.read(peer, len(payload) + 16) == payload
    finally:
        os.close(peer)


def test_record_sparse_4gib_fails_closed_without_reading(site):
    record = _dist_info(site) / "RECORD"
    _truncate_sparse(record, 4 << 30)
    with mock.patch.object(cap_shim.os, "read") as mock_read:
        line = serve_main.mlxlm_patch_status()
    mock_read.assert_not_called()
    assert "PARTIAL or unverified" in line
    assert "RECORD" in line


def test_record_exact_cap_succeeds_and_cap_plus_one_fails_closed(site):
    cap = serve_main.MLXLM_RECORD_MAX_BYTES
    record = _dist_info(site) / "RECORD"
    a_digest = _digest(b"A = 1\n")
    b_digest = _digest(b"B = 2\n")
    base_rows = (
        f"mlx_lm/a.py,{a_digest},6\n"
        f"mlx_lm/b.py,{b_digest},6\n"
    ).encode()
    # Pad with valid non-mlx_lm comment-like dummy paths to reach exact cap.
    row_line = b"other/" + b"x" * 90 + b",,\n"
    remaining = cap - len(base_rows)
    count, rem = divmod(remaining, len(row_line))
    if rem < 4:
        count -= 1
        rem += len(row_line)
    tail = b"z" * (rem - 3) + b",,\n"
    at_cap = base_rows + row_line * count + tail
    assert len(at_cap) == cap
    record.write_bytes(at_cap)
    assert "NOT applied" in serve_main.mlxlm_patch_status()

    record.write_bytes(at_cap + b"\n")
    assert len(record.read_bytes()) == cap + 1
    line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "RECORD" in line


def test_record_bad_utf8_dir_dangling_symlink_regular_symlink_and_fifo_distinfo(site, tmp_path):
    info = _dist_info(site)
    record = info / "RECORD"

    # Symlink to a regular RECORD works.
    real_rec = tmp_path / "real_record"
    real_rec.write_bytes(record.read_bytes())
    record.unlink()
    record.symlink_to(real_rec)
    assert "NOT applied" in serve_main.mlxlm_patch_status()

    # Dangling symlink fails closed (must not return None).
    record.unlink()
    record.symlink_to(tmp_path / "missing_record")
    assert "PARTIAL or unverified" in serve_main.mlxlm_patch_status()

    # Invalid UTF-8 RECORD fails closed.
    record.unlink()
    record.write_bytes(b"\xff\xfe\n")
    assert "PARTIAL or unverified" in serve_main.mlxlm_patch_status()

    # Directory named RECORD fails closed.
    record.unlink()
    record.mkdir()
    assert "PARTIAL or unverified" in serve_main.mlxlm_patch_status()
    record.rmdir()

    # Unreadable RECORD (chmod 000) fails closed.
    if os.geteuid() != 0:
        record.write_bytes(real_rec.read_bytes())
        record.chmod(0)
        try:
            assert "PARTIAL or unverified" in serve_main.mlxlm_patch_status()
        finally:
            record.chmod(0o600)

    # FIFO masquerading as dist-info path fails closed without hanging.
    fifo_dist = tmp_path / f"mlx_lm-{VERSION}.egg-info"
    os.mkfifo(fifo_dist)
    dist = importlib.metadata.PathDistribution(fifo_dist)
    with _watchdog(5), mock.patch.object(importlib.metadata, "distribution", return_value=dist):
        line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line


def test_record_duplicates_traversal_nul_and_quoted_csv(site):
    info = _dist_info(site)
    record = info / "RECORD"
    a_h = _digest(b"A = 1\n")
    b_h = _digest(b"B = 2\n")

    # Identical duplicate row is deduplicated (still 2 of 2, not 3).
    (site / "mlx_lm/a.py").write_bytes(b"A = 1  # patched\n")
    (site / "mlx_lm" / RECORD_NAME).write_text(f"series=patches\nmlx_lm={VERSION}\n")
    record.write_text(
        f"mlx_lm/a.py,{a_h},6\n"
        f"./mlx_lm/a.py,{a_h},6\n"
        f"mlx_lm/b.py,{b_h},6\n"
    )
    line = serve_main.mlxlm_patch_status()
    assert line.startswith("mlx-lm serve patches: applied (1 of 2")

    # Conflicting duplicate rows fail closed (even with alias ./mlx_lm/a.py).
    for first_spec in (b_h, "", "md5=deadbeef", "sha256"):
        record.write_text(
            f"mlx_lm/a.py,{first_spec},6\n"
            f"./mlx_lm/a.py,{a_h},6\n"
            f"mlx_lm/b.py,{b_h},6\n"
        )
        line = serve_main.mlxlm_patch_status()
        assert "PARTIAL or unverified" in line
        assert "applied (" not in line

    # Path traversal inside mlx_lm/ fails closed; standard ../../bin outside mlx_lm is allowed.
    record.write_text(
        f"../../bin/mlx_lm.generate,{a_h},6\n"
        f"mlx_lm/a.py,{a_h},6\n"
        f"mlx_lm/b.py,{b_h},6\n"
    )
    assert serve_main.mlxlm_patch_status().startswith("mlx-lm serve patches: applied (1 of 2")

    # Create escape.py outside mlx_lm/ matching a_h so traversal would falsely
    # verify if ".." were not rejected before opening.
    (site / "escape.py").write_bytes(b"A = 1\n")
    (site / "mlx_lm/sub").mkdir()
    real_open = os.open

    def _guard_escape(path, *args, **kwargs):
        if pathlib.Path(path).name == "escape.py":
            raise AssertionError("must not open traversal path escape.py")
        return real_open(path, *args, **kwargs)

    for bad_path in (
        "mlx_lm/../escape.py",
        "./mlx_lm/sub/../../escape.py",
        "mlx_lm/a.py/",
        "mlx_lm/a.py/.",
        "mlx_lm/a.py/./.",
    ):
        record.write_text(
            f"{bad_path},{a_h},6\n"
            f"mlx_lm/b.py,{b_h},6\n"
        )
        with mock.patch.object(cap_shim.os, "open", side_effect=_guard_escape):
            line = serve_main.mlxlm_patch_status()
        assert "PARTIAL or unverified" in line, bad_path
        assert "RECORD" in line, bad_path

    # Truncated, empty, garbage, unclosed-quote, wrong-column, empty-path,
    # bad-quote, or non-integer-size RECORD files must fail closed.
    corrupt_records = [
        "",
        "\n\n",
        "not a csv row\n",
        f'"mlx_lm/a.py,{a_h},6\nmlx_lm/b.py,{b_h},6\n',
        f"mlx_lm/a.py,{a_h}\nmlx_lm/b.py,{b_h},6\n",
        f"mlx_lm/a.py,{a_h},6,extra\nmlx_lm/b.py,{b_h},6\n",
        f",{a_h},6\nmlx_lm/b.py,{b_h},6\n",
        f"mlx_lm/a.py,{a_h},notanint\nmlx_lm/b.py,{b_h},6\n",
        f'"mlx_lm/"a.py,{a_h},6\nmlx_lm/b.py,{b_h},6\n',
    ]
    for bad_rec in corrupt_records:
        record.write_text(bad_rec)
        line = serve_main.mlxlm_patch_status()
        assert "PARTIAL or unverified" in line, repr(bad_rec)
        assert "RECORD" in line, repr(bad_rec)
        assert "applied (" not in line, repr(bad_rec)
        assert "NOT applied" not in line, repr(bad_rec)

    # Embedded NUL in RECORD fails closed.
    record.write_bytes(f"mlx_lm/a.py\x00,{a_h},6\nmlx_lm/b.py,{b_h},6\n".encode())
    assert "PARTIAL or unverified" in serve_main.mlxlm_patch_status()

    # Quoted CSV filename containing an embedded newline stays intact.
    nl_data = b"Z = 9\n"
    (site / "mlx_lm" / "line1\nline2.py").write_bytes(nl_data)
    record.write_text(
        f"mlx_lm/a.py,{a_h},6\n"
        f"mlx_lm/b.py,{b_h},6\n"
        f'"mlx_lm/line1\nline2.py",{_digest(nl_data)},{len(nl_data)}\n'
    )
    line = serve_main.mlxlm_patch_status()
    assert line.startswith("mlx-lm serve patches: applied (1 of 3")


# ---------------------------------------------- bounded METADATA/PKG-INFO reads


def test_metadata_fifo_does_not_block_or_fall_back_to_pkg_info(site):
    info = _dist_info(site)
    (site / "mlx_lm/a.py").write_bytes(b"A = 1  # patched\n")
    (site / "mlx_lm" / RECORD_NAME).write_text(f"series=patches\nmlx_lm={VERSION}\n")
    meta = info / "METADATA"
    meta.unlink()
    os.mkfifo(meta)
    (info / "PKG-INFO").write_text(f"Metadata-Version: 2.1\nName: mlx-lm\nVersion: {VERSION}\n")
    with _watchdog(5):
        line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "version could not be read" in line
    assert "applied (" not in line

    # A non-ENOENT lstat error on METADATA must also fail closed without
    # falling back to PKG-INFO.
    real_lstat = os.lstat

    def _eperm_meta(path, *args, **kwargs):
        if pathlib.Path(path) == meta:
            raise PermissionError("metadata lstat denied")
        return real_lstat(path, *args, **kwargs)

    with mock.patch.object(serve_main.os, "lstat", side_effect=_eperm_meta):
        line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "version could not be read" in line
    assert "applied (" not in line


def test_metadata_sparse_4gib_fails_closed_without_reading(site):
    info = _dist_info(site)
    (site / "mlx_lm/a.py").write_bytes(b"A = 1  # patched\n")
    (site / "mlx_lm" / RECORD_NAME).write_text(f"series=patches\nmlx_lm={VERSION}\n")
    meta = info / "METADATA"
    _truncate_sparse(meta, 4 << 30)
    real_read = os.read
    read_sparse = []

    def _guard_read(fd, n):
        if os.fstat(fd).st_size == (4 << 30):
            read_sparse.append(n)
        return real_read(fd, n)

    with mock.patch.object(cap_shim.os, "read", side_effect=_guard_read):
        line = serve_main.mlxlm_patch_status()
    assert read_sparse == []
    assert "PARTIAL or unverified" in line
    assert "version could not be read" in line


def test_metadata_exact_cap_succeeds_and_cap_plus_one_fails_closed(site):
    cap = serve_main.MLXLM_METADATA_MAX_BYTES
    info = _dist_info(site)
    (site / "mlx_lm/a.py").write_bytes(b"A = 1  # patched\n")
    (site / "mlx_lm" / RECORD_NAME).write_text(f"series=patches\nmlx_lm={VERSION}\n")
    header = f"Metadata-Version: 2.1\nName: mlx-lm\nVersion: {VERSION}\n\n".encode()
    meta = info / "METADATA"
    meta.write_bytes(header + b"x" * (cap - len(header)))
    assert serve_main.mlxlm_patch_status().startswith("mlx-lm serve patches: applied (1 of 2")

    meta.write_bytes(header + b"x" * (cap + 1 - len(header)))
    line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "version could not be read" in line


def test_metadata_bad_utf8_invalid_versions_and_pkg_info_fallback(site):
    info = _dist_info(site)
    (site / "mlx_lm/a.py").write_bytes(b"A = 1  # patched\n")
    (site / "mlx_lm" / RECORD_NAME).write_text(f"series=patches\nmlx_lm={VERSION}\n")
    meta = info / "METADATA"

    # Invalid UTF-8 METADATA
    meta.write_bytes(b"\xff\xfe\n")
    assert "version could not be read" in serve_main.mlxlm_patch_status()

    # Missing Version header
    meta.write_text("Metadata-Version: 2.1\nName: mlx-lm\n")
    assert "version could not be read" in serve_main.mlxlm_patch_status()

    # Duplicate Version headers
    meta.write_text(f"Metadata-Version: 2.1\nName: mlx-lm\nVersion: {VERSION}\nVersion: 0.0.1\n")
    assert "version could not be read" in serve_main.mlxlm_patch_status()

    # Folded multiline Version header
    meta.write_text(f"Metadata-Version: 2.1\nName: mlx-lm\nVersion: {VERSION}\n extra\n")
    assert "version could not be read" in serve_main.mlxlm_patch_status()

    # Deeply nested MIME multipart body (~108 KiB, well under 1 MiB cap) must
    # not trigger RecursionError in METADATA parsing.
    nested = "".join(
        f"Content-Type: multipart/mixed; boundary=b{i}\n\n--b{i}\n"
        for i in range(2000)
    )
    meta.write_text(f"Metadata-Version: 2.1\nName: mlx-lm\nVersion: {VERSION}\n{nested}")
    assert serve_main.mlxlm_patch_status().startswith("mlx-lm serve patches: applied (1 of 2")
    meta.write_text(f"Metadata-Version: 2.1\nName: mlx-lm\n{nested}")
    assert "version could not be read" in serve_main.mlxlm_patch_status()

    # Genuine PKG-INFO fallback when METADATA is absent
    meta.unlink()
    (info / "PKG-INFO").write_text(f"Metadata-Version: 2.1\nName: mlx-lm\nVersion: {VERSION}\n")
    assert serve_main.mlxlm_patch_status().startswith("mlx-lm serve patches: applied (1 of 2")


# ------------------------------------ bounded .omarchy-serve-patches reads


def test_patch_record_fifo_without_and_with_writer_fails_closed(site):
    (site / "mlx_lm/a.py").write_bytes(b"A = 1  # patched\n")
    rec = site / "mlx_lm" / RECORD_NAME
    os.mkfifo(rec)
    with _watchdog(5):
        line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "applied (" not in line

    peer = os.open(rec, os.O_RDWR | os.O_NONBLOCK)
    try:
        payload = f"series=patches\nmlx_lm={VERSION}\n".encode()
        os.write(peer, payload)
        with _watchdog(5):
            line = serve_main.mlxlm_patch_status()
        assert "PARTIAL or unverified" in line
        assert os.read(peer, len(payload) + 16) == payload
    finally:
        os.close(peer)


def test_patch_record_sparse_4gib_fails_closed_without_reading(site):
    (site / "mlx_lm/a.py").write_bytes(b"A = 1  # patched\n")
    rec = site / "mlx_lm" / RECORD_NAME
    _truncate_sparse(rec, 4 << 30)
    real_read = os.read
    read_sparse = []

    def _guard_read(fd, n):
        if os.fstat(fd).st_size == (4 << 30):
            read_sparse.append(n)
        return real_read(fd, n)

    with mock.patch.object(cap_shim.os, "read", side_effect=_guard_read):
        line = serve_main.mlxlm_patch_status()
    assert read_sparse == []
    assert "PARTIAL or unverified" in line
    assert "applied (" not in line


def test_patch_record_exact_cap_succeeds_and_cap_plus_one_fails_closed(site):
    cap = serve_main.MLXLM_PATCH_RECORD_MAX_BYTES
    (site / "mlx_lm/a.py").write_bytes(b"A = 1  # patched\n")
    rec = site / "mlx_lm" / RECORD_NAME
    tail = f"\nmlx_lm={VERSION}\n".encode()
    rec.write_bytes(b"#" * (cap - len(tail)) + tail)
    assert serve_main.mlxlm_patch_status().startswith("mlx-lm serve patches: applied (1 of 2")

    rec.write_bytes(b"#" * (cap + 1 - len(tail)) + tail)
    line = serve_main.mlxlm_patch_status()
    assert "PARTIAL or unverified" in line
    assert "applied (" not in line


def test_patch_record_bad_utf8_special_files_and_regular_symlink(site, tmp_path):
    (site / "mlx_lm/a.py").write_bytes(b"A = 1  # patched\n")
    rec = site / "mlx_lm" / RECORD_NAME

    # Invalid UTF-8 bytes
    for raw in (b"\xff\xfe\n", f"mlx_lm={VERSION}\n".encode() + b"\xc3"):
        rec.write_bytes(raw)
        line = serve_main.mlxlm_patch_status()
        assert "PARTIAL or unverified" in line
        assert "applied (" not in line

    # Directory
    rec.unlink()
    rec.mkdir()
    assert "PARTIAL or unverified" in serve_main.mlxlm_patch_status()
    rec.rmdir()

    # /dev/null symlink
    rec.symlink_to("/dev/null")
    assert "PARTIAL or unverified" in serve_main.mlxlm_patch_status()
    rec.unlink()

    # Symlink loop
    other = site / "mlx_lm/rec_loop"
    rec.symlink_to(other)
    other.symlink_to(rec)
    assert "PARTIAL or unverified" in serve_main.mlxlm_patch_status()
    rec.unlink()
    other.unlink()

    # Unix socket
    with tempfile.TemporaryDirectory(dir="/tmp") as sd:
        sock_path = pathlib.Path(sd) / "rec.sock"
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(str(sock_path))
            rec.symlink_to(sock_path)
            assert "PARTIAL or unverified" in serve_main.mlxlm_patch_status()
        finally:
            rec.unlink()
            sock.close()

    # Symlink to a regular completion record works.
    real_rec = tmp_path / "real_patch_record"
    real_rec.write_text(f"series=patches\nmlx_lm={VERSION}\n")
    rec.symlink_to(real_rec)
    assert serve_main.mlxlm_patch_status().startswith("mlx-lm serve patches: applied (1 of 2")


def test_patch_record_requires_exact_newline_terminated_line(site):
    (site / "mlx_lm/a.py").write_bytes(b"A = 1  # patched\n")
    rec = site / "mlx_lm" / RECORD_NAME
    for bad in (
        f"not_mlx_lm={VERSION}\n",
        f"# mlx_lm={VERSION}\n",
        f"series=patches\nmlx_lm={VERSION}",
        f"series=patches\nmlx_lm={VERSION} \n",
    ):
        rec.write_text(bad)
        line = serve_main.mlxlm_patch_status()
        assert "PARTIAL or unverified" in line, repr(bad)
        assert "applied (" not in line, repr(bad)

    rec.write_text(f"mlx_lm={VERSION}\n")
    assert serve_main.mlxlm_patch_status().startswith("mlx-lm serve patches: applied (1 of 2")
