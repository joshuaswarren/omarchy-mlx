"""Bounded regular-file I/O, manifest validation, clip caps, and atomic output
writes for serve/mlx_omarchy_assistant/wake_eval.py.

Every input is finite: oversized files are sparse or run against a lowered cap,
and each FIFO test runs under a SIGALRM watchdog so a regression fails instead
of hanging the test run.
"""
from __future__ import annotations

import contextlib
import errno
import io
import json
import os
import signal
import struct
import sys
import tempfile
import tracemalloc
import types
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import wake_eval  # noqa: E402

DEEP_JSON = b"[" * 200_000 + b"]" * 200_000
TOO_DEEP_JSON = b'[{"file": "a.wav", "label": "positive", "meta": ' + b"[" * 40 + b"]" * 40 + b"}]"


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


def write_native_wav(path: Path, seconds: float = 0.2) -> Path:
    rate = wake_eval.SAMPLE_RATE
    frames = np.zeros(int(seconds * rate), dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())
    return path


def write_wav_header_only(
    path: Path,
    *,
    nchannels: int = 1,
    sampwidth: int = 2,
    framerate: int = wake_eval.SAMPLE_RATE,
    data_bytes: int,
    truncate_to_data: bool = False,
) -> Path:
    riff_size = min(0xFFFFFFFF, 36 + data_bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        nchannels,
        framerate,
        framerate * nchannels * sampwidth,
        nchannels * sampwidth,
        sampwidth * 8,
        b"data",
        data_bytes,
    )
    with open(path, "wb") as fh:
        fh.write(header)
        if truncate_to_data:
            fh.truncate(len(header) + data_bytes)
    return path


class ScriptedSession:
    def predict(self, pcm):
        rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
        return {"hey_jarvis_v0.1": 0.9 if rms > 1000 else 0.1}

    def reset(self):
        pass


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
    with redirect_stdout(out), redirect_stderr(err):
        rc = wake_eval.main(argv, session=ScriptedSession())
    return rc, out.getvalue(), err.getvalue()


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)


class BoundedRegularReaderTests(TempDirCase):
    def test_reads_regular_file_up_to_exact_cap(self) -> None:
        p = self.root / "ok.txt"
        p.write_bytes(b"a" * 64)
        self.assertEqual(wake_eval._read_regular_bytes(p, 64), b"a" * 64)
        self.assertEqual(wake_eval._read_regular_text(p, 64), "a" * 64)

    def test_follows_symlink_to_regular_file(self) -> None:
        real = self.root / "real.txt"
        real.write_text("hello\n", encoding="utf-8")
        link = self.root / "link.txt"
        link.symlink_to(real)
        self.assertEqual(wake_eval._read_regular_text(link, 64), "hello\n")

    def test_refuses_fifo_without_blocking(self) -> None:
        pipe = make_fifo(self.root / "pipe")
        with watchdog(), self.assertRaisesRegex(wake_eval.WakeEvalError, "non-regular"):
            wake_eval._read_regular_bytes(pipe, 64)

    def test_refuses_symlink_to_fifo_without_blocking(self) -> None:
        pipe = make_fifo(self.root / "pipe")
        link = self.root / "link"
        link.symlink_to(pipe)
        with watchdog(), self.assertRaisesRegex(wake_eval.WakeEvalError, "non-regular"):
            wake_eval._read_regular_bytes(link, 64)

    def test_refuses_directory_and_device(self) -> None:
        with self.assertRaisesRegex(wake_eval.WakeEvalError, "non-regular"):
            wake_eval._read_regular_bytes(self.root, 64)
        devnull = Path("/dev/null")
        if devnull.exists():
            with self.assertRaisesRegex(wake_eval.WakeEvalError, "non-regular"):
                wake_eval._read_regular_bytes(devnull, 64)

    def test_refuses_oversized_file_before_any_read(self) -> None:
        big = sparse_file(self.root / "big.bin", 1025)
        with (
            mock.patch.object(
                wake_eval,
                "_read_bounded_chunks",
                side_effect=AssertionError("read before st_size check"),
            ),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "exceeds 1024 bytes"),
        ):
            wake_eval._read_regular_bytes(big, 1024)

    @unittest.skipUnless(Path("/proc/self/status").exists(), "needs procfs")
    def test_understated_st_size_is_capped_by_chunked_read(self) -> None:
        # procfs files report st_size == 0 while holding far more than 8 bytes.
        with self.assertRaisesRegex(wake_eval.WakeEvalError, "exceeds 8 bytes"):
            wake_eval._read_regular_bytes(Path("/proc/self/status"), 8)

    def test_refuses_invalid_utf8(self) -> None:
        bad = self.root / "bad.txt"
        bad.write_bytes(b"ok\xff")
        with self.assertRaisesRegex(wake_eval.WakeEvalError, "not valid UTF-8"):
            wake_eval._read_regular_text(bad, 64)

    def test_closes_fd_on_every_open_regular_error_path(self) -> None:
        p = self.root / "f.bin"
        p.write_bytes(b"0123456789")
        closed: list[int] = []
        real_close = os.close

        def track_close(fd: int) -> None:
            closed.append(fd)
            real_close(fd)

        # 1. fstat failure
        with (
            mock.patch.object(wake_eval.os, "close", side_effect=track_close),
            mock.patch.object(wake_eval.os, "fstat", side_effect=OSError(errno.EIO, "io")),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "cannot read"),
        ):
            wake_eval._read_regular_bytes(p, 64)
        self.assertEqual(len(closed), 1)

        # 2. non-regular file
        pipe = make_fifo(self.root / "p.fifo")
        with (
            watchdog(),
            mock.patch.object(wake_eval.os, "close", side_effect=track_close),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "non-regular"),
        ):
            wake_eval._read_regular_bytes(pipe, 64)
        self.assertEqual(len(closed), 2)

        # 3. oversized st_size
        with (
            mock.patch.object(wake_eval.os, "close", side_effect=track_close),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "exceeds 4 bytes"),
        ):
            wake_eval._read_regular_bytes(p, 4)
        self.assertEqual(len(closed), 3)

        # 4. fdopen failure
        with (
            mock.patch.object(wake_eval.os, "close", side_effect=track_close),
            mock.patch.object(wake_eval.os, "fdopen", side_effect=RuntimeError("boom")),
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            wake_eval._read_regular_bytes(p, 64)
        self.assertEqual(len(closed), 4)


class CorpusListBoundsTests(TempDirCase):
    def test_valid_corpus_list_uses_bounded_read(self) -> None:
        clip = write_native_wav(self.root / "a.wav")
        lst = self.root / "corpus.txt"
        lst.write_text(f"\n  {clip}  \n\n", encoding="utf-8")
        out = self.root / "out.json"
        with forbid_path_io():
            rc, _stdout, stderr = run_cli([
                "--mode", "corpus",
                "--corpus-list", str(lst),
                "--model", "unused",
                "--thresholds", "0.5",
                "--floors", "0",
                "--out", str(out),
            ])
        self.assertEqual(rc, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(out.read_text())["rows"][0]["files"], 1)

    def test_corpus_list_fifo_refused_cleanly_in_cli(self) -> None:
        pipe = make_fifo(self.root / "corpus.fifo")
        out = self.root / "out.json"
        with watchdog():
            rc, _stdout, stderr = run_cli([
                "--mode", "corpus",
                "--corpus-list", str(pipe),
                "--model", "unused",
                "--out", str(out),
            ])
        self.assertEqual(rc, 1)
        self.assertIn("wake_eval:", stderr)
        self.assertIn("non-regular", stderr)
        self.assertFalse(out.exists())

    def test_corpus_list_oversized_refused_cleanly(self) -> None:
        big = sparse_file(self.root / "corpus.txt", wake_eval.MAX_CORPUS_LIST_BYTES + 1)
        with self.assertRaisesRegex(wake_eval.WakeEvalError, "exceeds"):
            wake_eval.load_corpus_list(big)
        out = self.root / "out.json"
        rc, _stdout, stderr = run_cli([
            "--mode", "corpus",
            "--corpus-list", str(big),
            "--model", "unused",
            "--out", str(out),
        ])
        self.assertEqual(rc, 1)
        self.assertIn("wake_eval:", stderr)
        self.assertIn("exceeds", stderr)

    def test_corpus_list_nul_byte_refused(self) -> None:
        bad = self.root / "nul.txt"
        bad.write_bytes(b"a.wav\x00b.wav\n")
        with self.assertRaisesRegex(wake_eval.WakeEvalError, "NUL"):
            wake_eval.load_corpus_list(bad)


class ManifestBoundsAndValidationTests(TempDirCase):
    def test_valid_manifest_uses_bounded_read_and_atomic_write(self) -> None:
        write_native_wav(self.root / "pos.wav")
        write_native_wav(self.root / "neg.wav")
        m = self.root / "m.json"
        m.write_text(json.dumps([
            {"file": "pos.wav", "label": "positive"},
            {"file": "neg.wav", "label": "negative"},
        ]), encoding="utf-8")
        out = self.root / "results.json"
        with forbid_path_io():
            rc, _stdout, stderr = run_cli([
                "--manifest", str(m),
                "--clips-dir", str(self.root),
                "--model", "unused",
                "--thresholds", "0.5",
                "--floors", "0",
                "--out", str(out),
            ])
        self.assertEqual(rc, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(out.read_text())["rows"][0]["tn"], 1)

    def test_manifest_fifo_refused_without_blocking(self) -> None:
        pipe = make_fifo(self.root / "m.fifo")
        out = self.root / "out.json"
        with watchdog():
            rc, _stdout, stderr = run_cli([
                "--manifest", str(pipe),
                "--clips-dir", str(self.root),
                "--model", "unused",
                "--out", str(out),
            ])
        self.assertEqual(rc, 1)
        self.assertIn("wake_eval:", stderr)
        self.assertIn("non-regular", stderr)

    def test_manifest_oversized_refused(self) -> None:
        big = sparse_file(self.root / "m.json", wake_eval.MAX_MANIFEST_BYTES + 1)
        with self.assertRaisesRegex(wake_eval.WakeEvalError, "exceeds"):
            wake_eval.load_manifest(big)

    def test_manifest_deep_nesting_and_recursion_error_refused(self) -> None:
        deep = self.root / "deep.json"
        deep.write_bytes(DEEP_JSON)
        # Accept either parser RecursionError refusal or depth-cap refusal across Python versions.
        with self.assertRaisesRegex(
            wake_eval.WakeEvalError,
            r"unreadable|nests deeper than 32 levels",
        ):
            wake_eval.load_manifest(deep)

        too_deep = self.root / "too_deep.json"
        too_deep.write_bytes(TOO_DEEP_JSON)
        with self.assertRaisesRegex(
            wake_eval.WakeEvalError,
            "nests deeper than 32 levels",
        ):
            wake_eval.load_manifest(too_deep)

        # Exact boundary: outer list (1) -> dict (2) -> 30 nested lists (32) passes;
        # 31 nested lists (33) is refused.
        at_limit = self.root / "depth32.json"
        at_limit.write_bytes(
            b'[{"file": "a.wav", "label": "positive", "meta": '
            + b"[" * 30 + b"]" * 30 + b"}]"
        )
        self.assertEqual(len(wake_eval.load_manifest(at_limit)), 1)

        over_limit = self.root / "depth33.json"
        over_limit.write_bytes(
            b'[{"file": "a.wav", "label": "positive", "meta": '
            + b"[" * 31 + b"]" * 31 + b"}]"
        )
        with self.assertRaisesRegex(
            wake_eval.WakeEvalError,
            "nests deeper than 32 levels",
        ):
            wake_eval.load_manifest(over_limit)

        # A JSON null sibling before a deep container must not truncate traversal.
        after_null = self.root / "after_null.json"
        after_null.write_bytes(
            b'[{"file": "a.wav", "label": "positive", "note": null, "meta": '
            + b"[" * 31 + b"]" * 31 + b"}]"
        )
        with self.assertRaisesRegex(
            wake_eval.WakeEvalError,
            "nests deeper than 32 levels",
        ):
            wake_eval.load_manifest(after_null)

        # Deterministic RecursionError coverage regardless of C stack size.
        valid = self.root / "valid.json"
        valid.write_text('[{"file": "a.wav", "label": "positive"}]', encoding="utf-8")
        with (
            mock.patch.object(
                wake_eval.json,
                "loads",
                side_effect=RecursionError("maximum recursion depth exceeded"),
            ),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "unreadable"),
        ):
            wake_eval.load_manifest(valid)

    def test_manifest_shape_validated_before_opening_any_clips(self) -> None:
        bad_cases = (
            ("dict-root", '{"file": "a.wav", "label": "positive"}', "must be a JSON list"),
            ("scalar-root", '"hello"', "must be a JSON list"),
            ("non-dict-entry", '["a.wav"]', "entry 0 must be a JSON object"),
            ("missing-file", '[{"label": "positive"}]', "non-empty string 'file'"),
            ("empty-file", '[{"file": "  ", "label": "positive"}]', "non-empty string 'file'"),
            ("int-file", '[{"file": 123, "label": "positive"}]', "non-empty string 'file'"),
            ("nul-file", '[{"file": "a\\u0000.wav", "label": "positive"}]', "non-empty string 'file'"),
            ("surrogate-file", '[{"file": "a.wav", "label": "positive"}, {"file": "\\ud800.wav", "label": "negative"}]', "entry 1 has unencodable 'file'"),
            ("missing-label", '[{"file": "a.wav"}]', "'label'"),
            ("bad-label", '[{"file": "a.wav", "label": "maybe"}]', "'label'"),
            ("int-label", '[{"file": "a.wav", "label": 1}]', "'label'"),
            ("second-entry-bad", '[{"file": "a.wav", "label": "positive"}, {"file": "b.wav", "label": "bad"}]', "entry 1"),
            ("syntax-error", "{not json", "unreadable"),
        )
        out = self.root / "out.json"
        for name, body, pattern in bad_cases:
            with self.subTest(name):
                m = self.root / f"{name}.json"
                m.write_text(body, encoding="utf-8")
                with mock.patch.object(
                    wake_eval,
                    "load_pcm",
                    side_effect=AssertionError("opened clip before manifest validation"),
                ):
                    rc, _stdout, stderr = run_cli([
                        "--manifest", str(m),
                        "--clips-dir", str(self.root),
                        "--model", "unused",
                        "--out", str(out),
                    ])
                self.assertEqual(rc, 1)
                self.assertIn("wake_eval:", stderr)
                self.assertRegex(stderr, pattern)
                self.assertFalse(out.exists())


class LoadPcmBoundsTests(TempDirCase):
    def test_fifo_clip_refused_without_blocking(self) -> None:
        pipe = make_fifo(self.root / "clip.wav")
        with watchdog(), self.assertRaisesRegex(wake_eval.WakeEvalError, "non-regular"):
            wake_eval.load_pcm(pipe)

    def test_directory_clip_refused(self) -> None:
        d = self.root / "dir.wav"
        d.mkdir()
        with self.assertRaisesRegex(wake_eval.WakeEvalError, "non-regular"):
            wake_eval.load_pcm(d)

    def test_symlink_to_regular_wav_loads_and_symlink_to_fifo_refused(self) -> None:
        real = write_native_wav(self.root / "real.wav")
        link = self.root / "link.wav"
        link.symlink_to(real)
        pcm = wake_eval.load_pcm(link)
        self.assertEqual(len(pcm), int(wake_eval.MIN_SECONDS * wake_eval.SAMPLE_RATE) * 2)

        pipe = make_fifo(self.root / "pipe.wav")
        bad_link = self.root / "bad_link.wav"
        bad_link.symlink_to(pipe)
        with watchdog(), self.assertRaisesRegex(wake_eval.WakeEvalError, "non-regular"):
            wake_eval.load_pcm(bad_link)

    def test_oversized_sparse_clip_refused_before_wave_open(self) -> None:
        big = write_wav_header_only(
            self.root / "huge.wav",
            data_bytes=wake_eval.MAX_CLIP_BYTES + 1024,
            truncate_to_data=True,
        )
        with (
            mock.patch.object(
                wave, "open", side_effect=AssertionError("wave.open called on oversized clip")
            ),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "exceeds"),
        ):
            wake_eval.load_pcm(big)

    def test_inflated_wav_header_refused_before_reading_frames(self) -> None:
        # 44-byte file claiming > MAX_PCM_BYTES of decoded native PCM.
        lying = write_wav_header_only(
            self.root / "lying.wav",
            data_bytes=wake_eval.MAX_PCM_BYTES + 4096,
            truncate_to_data=False,
        )
        with (
            mock.patch.object(
                wake_eval,
                "_read_wav_pcm",
                side_effect=AssertionError("readframes called on inflated WAV header"),
            ),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "decoded PCM exceeds"),
        ):
            wake_eval.load_pcm(lying)

    def test_inflated_non_native_wav_header_refused_before_resample(self) -> None:
        lying = write_wav_header_only(
            self.root / "lying48k.wav",
            nchannels=2,
            sampwidth=2,
            framerate=48000,
            data_bytes=wake_eval.MAX_PCM_BYTES + 4096,
            truncate_to_data=False,
        )
        with (
            mock.patch.object(
                wake_eval,
                "_resample_stream",
                side_effect=AssertionError("resample called on inflated non-native WAV"),
            ),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "decoded PCM exceeds"),
        ):
            wake_eval.load_pcm(lying)

    def test_odd_byte_native_wav_data_chunk_refused(self) -> None:
        # 5-byte data chunk on 16-bit mono WAV: wave.getnframes() rounds down to 2 frames (4 bytes).
        odd = write_wav_header_only(
            self.root / "odd.wav",
            nchannels=1,
            sampwidth=2,
            framerate=wake_eval.SAMPLE_RATE,
            data_bytes=5,
            truncate_to_data=True,
        )
        with self.assertRaisesRegex(wake_eval.WakeEvalError, "misaligned WAV data chunk"):
            wake_eval.load_pcm(odd)

    def test_wav_exceeding_max_chunks_before_data_refused(self) -> None:
        # 1,025 JUNK chunks before fmt/data must fail explicitly rather than bypassing
        # the alignment check.
        fmt_chunk = struct.pack(
            "<4sIHHIIHH",
            b"fmt ", 16, 1, 1, wake_eval.SAMPLE_RATE, wake_eval.SAMPLE_RATE * 2, 2, 16,
        )
        junk = b"JUNK\x00\x00\x00\x00" * (wake_eval._MAX_WAV_CHUNKS + 1)
        data_chunk = b"data" + struct.pack("<I", 5) + b"\x00" * 5
        body = b"WAVE" + junk + fmt_chunk + data_chunk
        path = self.root / "many_chunks.wav"
        path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
        with self.assertRaisesRegex(wake_eval.WakeEvalError, "exceeds 1024 chunks"):
            wake_eval.load_pcm(path)

    def test_truncated_header_under_cap_refused_without_large_allocation(self) -> None:
        # 144-byte WAV declaring full 256 MiB of native PCM must fail before
        # _read_wav_pcm allocates anything for the missing payload.
        lying = write_wav_header_only(
            self.root / "lying_under_cap.wav",
            data_bytes=wake_eval.MAX_PCM_BYTES,
            truncate_to_data=False,
        )
        with open(lying, "ab") as fh:
            fh.write(b"\x00" * 100)
        with (
            mock.patch.object(
                wake_eval,
                "_read_wav_pcm",
                side_effect=AssertionError("_read_wav_pcm called when header exceeds file size"),
            ),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "truncated WAV data"),
        ):
            wake_eval.load_pcm(lying)

    def test_non_native_resample_uses_single_contiguous_input_buffer(self) -> None:
        # Multi-block 48 kHz mono stream: 300,000 frames (~2.4 MB float64 input,
        # 100,000 frames / ~0.8 MB float64 output). Peak overhead above input +
        # output must stay below a second input copy (no chunk-list + concat).
        clip = self.root / "contig48k.flac"
        clip.write_bytes(b"fLaC" + bytes(64))
        total_frames = 300_000
        block_size = 60_000

        class MultiBlock48k:
            def __init__(self, f):
                self.channels = 1
                self.samplerate = 48000
                self.frames = total_frames
                self._left = total_frames

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, frames=-1, dtype="float64", always_2d=True):
                if self._left <= 0:
                    return np.empty((0, 1), dtype=np.float64)
                n = min(self._left, block_size)
                self._left -= n
                return np.full((n, 1), 0.125, dtype=np.float64)

        def stub_resample(x, up, down):
            return np.asarray(x[::3], dtype=np.float64).copy()

        fake_sf = types.SimpleNamespace(SoundFile=MultiBlock48k)
        fake_sig = types.SimpleNamespace(resample_poly=stub_resample)
        with mock.patch.dict(
            sys.modules,
            {"soundfile": fake_sf, "scipy": types.SimpleNamespace(signal=fake_sig), "scipy.signal": fake_sig},
        ):
            tracemalloc.start()
            try:
                pcm = wake_eval.load_pcm(clip)
                _cur, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        self.assertEqual(len(pcm), (total_frames // 3) * 2)
        input_bytes = total_frames * 8
        output_bytes = (total_frames // 3) * 8
        # Single contiguous buffer keeps overhead well below a second input copy
        # (chunk-list + concatenate peaks at ~2*input + output = ~5.6 MB+).
        self.assertLess(peak - (input_bytes + output_bytes), input_bytes)

    def test_non_native_resample_grows_buffer_when_header_understates_frames(self) -> None:
        # 48 kHz stream whose header declares 600 frames while yielding 5 blocks
        # of 600 frames each; lowering _WAV_CHUNK_FRAMES to 600 forces resizing
        # a populated buffer across blocks and verifies earlier samples survive.
        clip = self.root / "grow48k.flac"
        clip.write_bytes(b"fLaC" + bytes(64))
        blocks = [
            np.full((600, 2), v, dtype=np.float64)
            for v in (0.125, 0.25, -0.25, 0.5, -0.5)
        ]

        class Understated48k:
            def __init__(self, f):
                self.channels = 2
                self.samplerate = 48000
                self.frames = 600
                self._iter = iter(blocks)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, frames=-1, dtype="float64", always_2d=True):
                return next(self._iter, np.empty((0, 2), dtype=np.float64))

        def stub_resample(x, up, down):
            return np.asarray(x[::3], dtype=np.float64)

        fake_sf = types.SimpleNamespace(SoundFile=Understated48k)
        fake_sig = types.SimpleNamespace(resample_poly=stub_resample)
        with (
            mock.patch.object(wake_eval, "MIN_SECONDS", 0.0),
            mock.patch.object(wake_eval, "_WAV_CHUNK_FRAMES", 600),
            mock.patch.dict(
                sys.modules,
                {"soundfile": fake_sf, "scipy": types.SimpleNamespace(signal=fake_sig), "scipy.signal": fake_sig},
            ),
        ):
            out = np.frombuffer(wake_eval.load_pcm(clip), dtype="<i2")
        expected = np.concatenate([
            np.full(200, int(round(v * 32768.0)), dtype="<i2")
            for v in (0.125, 0.25, -0.25, 0.5, -0.5)
        ])
        self.assertTrue(np.array_equal(out, expected))

    def test_production_soundfile_decoder_matches_mixdown_resample_and_clipping(self) -> None:
        # Runs stereo 16 kHz mixdown, stereo 48 kHz mixdown+resample, overshoot
        # clipping, and float scaling directly through the production SoundFile(fh)
        # streaming path (without sf.read).
        import math as _math

        def run_through_soundfile(
            frames_arr: np.ndarray, sr: int, *, overshoot: float = 1.0
        ) -> np.ndarray:
            clip = self.root / f"sf_{sr}_{overshoot}.flac"
            clip.write_bytes(b"fLaC" + bytes(64))

            class StreamDouble:
                def __init__(self, fh):
                    assert hasattr(fh, "fileno")
                    self.channels = frames_arr.shape[1]
                    self.samplerate = sr
                    self.frames = len(frames_arr)
                    self._pos = 0

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self, frames=-1, dtype="float64", always_2d=True):
                    if self._pos >= len(frames_arr):
                        return np.empty((0, self.channels), dtype=np.float64)
                    step = len(frames_arr) if frames < 0 else min(frames, 1024)
                    chunk = frames_arr[self._pos:self._pos + step]
                    self._pos += len(chunk)
                    return chunk

            def resample_poly(x, up, down):
                g = _math.gcd(up, down)
                return np.asarray(x, dtype=np.float64)[::down // g] * overshoot

            fake_sf = types.SimpleNamespace(SoundFile=StreamDouble)
            fake_sig = types.SimpleNamespace(resample_poly=resample_poly)
            with mock.patch.dict(
                sys.modules,
                {"soundfile": fake_sf, "scipy": types.SimpleNamespace(signal=fake_sig), "scipy.signal": fake_sig},
            ):
                return np.frombuffer(wake_eval.load_pcm(clip), dtype="<i2")

        n16 = int(2.5 * wake_eval.SAMPLE_RATE)
        stereo16 = np.stack(
            [np.full(n16, 8000 / 32768.0), np.full(n16, -2000 / 32768.0)], axis=1
        )
        out16 = run_through_soundfile(stereo16, wake_eval.SAMPLE_RATE)
        self.assertEqual(len(out16), n16)
        self.assertTrue(np.all(out16 == 3000))

        n48 = int(2.5 * 48000)
        stereo48 = np.stack(
            [np.full(n48, 6000 / 32768.0), np.full(n48, 2000 / 32768.0)], axis=1
        )
        out48 = run_through_soundfile(stereo48, 48000)
        self.assertEqual(len(out48), int(2.5 * wake_eval.SAMPLE_RATE))
        self.assertTrue(np.all(out48 == 4000))

        edge48 = np.where(np.arange(n48) < n48 // 2, 32767 / 32768.0, -1.0)[:, None]
        out_over = run_through_soundfile(edge48, 48000, overshoot=1.09)
        half = len(out_over) // 2
        self.assertTrue(np.all(out_over[:half] == 32767))
        self.assertTrue(np.all(out_over[half:] == -32768))

        n1 = wake_eval.SAMPLE_RATE
        float_src = np.repeat([0.5, 1.5, -1.5], n1)[:, None]
        out_flt = run_through_soundfile(float_src, wake_eval.SAMPLE_RATE)
        self.assertEqual(len(out_flt), 3 * n1)
        self.assertTrue(np.all(out_flt[:n1] == 16384))
        self.assertTrue(np.all(out_flt[n1:2 * n1] == 32767))
        self.assertTrue(np.all(out_flt[2 * n1:] == -32768))

    def test_non_native_16k_clip_fitting_int16_cap_loads_without_float64_rejection(self) -> None:
        # 16 kHz non-native clip (e.g. FLAC): 400 frames = 800 int16 bytes <= 1024,
        # while 400 * 8 = 3200 > 1024. Must load cleanly via blockwise int16 conversion.
        clip = self.root / "long16k.flac"
        clip.write_bytes(b"fLaC" + bytes(64))

        class Stream16k:
            def __init__(self, f):
                self.channels = 1
                self.samplerate = wake_eval.SAMPLE_RATE
                self.frames = 400
                self._left = 400

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, frames=-1, dtype="float64", always_2d=True):
                if self._left <= 0:
                    return np.empty((0, 1), dtype=np.float64)
                n = min(self._left, 100)
                self._left -= n
                return np.full((n, 1), 0.25, dtype=np.float64)

        fake_sf = types.SimpleNamespace(SoundFile=Stream16k)
        with (
            mock.patch.object(wake_eval, "MAX_PCM_BYTES", 1024),
            mock.patch.object(wake_eval, "MAX_RESAMPLE_WORK_BYTES", 1024),
            mock.patch.object(wake_eval, "MIN_SECONDS", 0.0),
            mock.patch.dict(sys.modules, {"soundfile": fake_sf}),
        ):
            pcm = wake_eval.load_pcm(clip)
        self.assertEqual(len(pcm), 800)
        self.assertTrue(np.all(np.frombuffer(pcm, dtype="<i2") == 8192))

    def test_resample_workspace_float64_output_cap_refused_before_scipy(self) -> None:
        # 8000 Hz -> 16000 Hz: 100 input frames -> 200 output frames.
        # int16 output = 400 bytes <= MAX_PCM_BYTES (2048),
        # input float64 = 800 bytes <= 1024,
        # output float64 = 1600 bytes > MAX_RESAMPLE_WORK_BYTES (1024) -> refused before SciPy.
        clip = self.root / "upsample8k.flac"
        clip.write_bytes(b"fLaC" + bytes(64))

        class Stream8k:
            def __init__(self, f):
                self.channels = 1
                self.samplerate = 8000
                self.frames = 100

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, frames=-1, dtype="float64", always_2d=True):
                raise AssertionError("read called past resample workspace cap")

        fake_sf = types.SimpleNamespace(SoundFile=Stream8k)
        fake_sig = types.SimpleNamespace(
            resample_poly=mock.Mock(side_effect=AssertionError("resample_poly called past float64 cap"))
        )
        with (
            mock.patch.object(wake_eval, "MAX_PCM_BYTES", 2048),
            mock.patch.object(wake_eval, "MAX_RESAMPLE_WORK_BYTES", 1024),
            mock.patch.dict(
                sys.modules,
                {"soundfile": fake_sf, "scipy": types.SimpleNamespace(signal=fake_sig), "scipy.signal": fake_sig},
            ),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "resampling workspace exceeds 1024 bytes"),
        ):
            wake_eval.load_pcm(clip)

        # Same workspace cap on a 48 kHz non-native WAV header (input float64 > cap while int16 fits).
        wav48 = write_wav_header_only(
            self.root / "work48k.wav",
            nchannels=1,
            sampwidth=2,
            framerate=48000,
            data_bytes=400,  # 200 frames -> 400 int16 bytes <= 2048, 1600 float64 bytes > 1024
            truncate_to_data=True,
        )
        with (
            mock.patch.object(wake_eval, "MAX_PCM_BYTES", 2048),
            mock.patch.object(wake_eval, "MAX_RESAMPLE_WORK_BYTES", 1024),
            mock.patch.object(
                wake_eval,
                "_resample_stream",
                side_effect=AssertionError("resample called past float64 workspace cap"),
            ),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "resampling workspace exceeds 1024 bytes"),
        ):
            wake_eval.load_pcm(wav48)

    def test_native_wav_passes_open_file_object_to_wave_open(self) -> None:
        clip = write_native_wav(self.root / "native.wav")
        seen_args: list[object] = []
        real_wave_open = wave.open

        def spy_wave_open(f, mode=None):
            seen_args.append(f)
            return real_wave_open(f, mode)

        with mock.patch.object(wave, "open", side_effect=spy_wave_open):
            wake_eval.load_pcm(clip)
        self.assertEqual(len(seen_args), 1)
        self.assertNotIsInstance(seen_args[0], (str, bytes, os.PathLike))
        self.assertTrue(hasattr(seen_args[0], "fileno"))

    def test_read_wav_pcm_enforces_cap_even_if_readframes_overruns(self) -> None:
        clip = write_native_wav(self.root / "small.wav", seconds=0.01)
        fake_handle = mock.MagicMock()
        fake_handle.getnchannels.return_value = 1
        fake_handle.getsampwidth.return_value = 2
        fake_handle.getframerate.return_value = wake_eval.SAMPLE_RATE
        fake_handle.getnframes.return_value = 10
        fake_handle.readframes.return_value = b"\x00" * 128
        fake_ctx = mock.MagicMock()
        fake_ctx.__enter__.return_value = fake_handle
        fake_ctx.__exit__.return_value = False
        with (
            mock.patch.object(wake_eval, "MAX_PCM_BYTES", 64),
            mock.patch.object(wave, "open", return_value=fake_ctx),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "decoded PCM exceeds 64 bytes"),
        ):
            wake_eval.load_pcm(clip)

    def test_truncated_zero_byte_clip_refused(self) -> None:
        empty = self.root / "empty.wav"
        empty.write_bytes(b"")
        with self.assertRaisesRegex(wake_eval.WakeEvalError, "truncated or unreadable"):
            wake_eval.load_pcm(empty)

    def test_truncated_native_wav_payload_refused_before_padding(self) -> None:
        # Valid 44-byte header declaring 1600 frames (3200 bytes), but only 100 bytes present.
        short = write_wav_header_only(
            self.root / "truncated.wav",
            data_bytes=3200,
            truncate_to_data=False,
        )
        with open(short, "ab") as fh:
            fh.write(b"\x00" * 100)
        with self.assertRaisesRegex(wake_eval.WakeEvalError, "truncated WAV data"):
            wake_eval.load_pcm(short)

        # Also test a truncated data chunk when file size exceeds declared data
        # (because of a preceding JUNK chunk), exercising _read_wav_pcm's EOF check.
        fmt_chunk = struct.pack(
            "<4sIHHIIHH",
            b"fmt ", 16, 1, 1, wake_eval.SAMPLE_RATE, wake_eval.SAMPLE_RATE * 2, 2, 16,
        )
        junk = b"JUNK" + struct.pack("<I", 4000) + b"\x00" * 4000
        data_chunk = b"data" + struct.pack("<I", 3200) + b"\x00" * 100
        body = b"WAVE" + junk + fmt_chunk + data_chunk
        after_junk = self.root / "truncated_after_junk.wav"
        after_junk.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
        with self.assertRaisesRegex(wake_eval.WakeEvalError, "expected 3200 bytes, got 100"):
            wake_eval.load_pcm(after_junk)

    def test_wave_open_runtime_error_wrapped_as_wake_eval_error(self) -> None:
        clip = write_native_wav(self.root / "ok.wav")
        with (
            mock.patch.object(wave, "open", side_effect=RuntimeError("corrupt chunk")),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "truncated or unreadable"),
        ):
            wake_eval.load_pcm(clip)

    def test_soundfile_fallback_uses_original_open_descriptor_not_path(self) -> None:
        clip = self.root / "clip.flac"
        payload = b"fLaC" + b"\x11\x22\x33\x44" * 16
        clip.write_bytes(payload)
        seen_bytes: list[bytes] = []
        seen_file_obj: list[object] = []

        class FakeSoundFile:
            def __init__(self, f):
                seen_file_obj.append(f)
                self.channels = 1
                self.samplerate = wake_eval.SAMPLE_RATE
                self.frames = 100
                self._done = False
                # Verify descriptor was rewound to 0 and holds original bytes even
                # after the path on disk was replaced by a FIFO.
                seen_bytes.append(f.read())

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, frames=-1, dtype="float64", always_2d=True):
                if self._done:
                    return np.empty((0, 1), dtype=np.float64)
                self._done = True
                return np.full((100, 1), 0.25, dtype=np.float64)

        real_wave_open = wave.open

        def swap_to_fifo_during_probe(f, mode=None):
            # Replace the pathname with a FIFO after _open_regular opened it.
            clip.unlink()
            os.mkfifo(clip)
            return real_wave_open(f, mode)

        fake_sf = types.SimpleNamespace(
            SoundFile=FakeSoundFile,
            read=mock.Mock(side_effect=AssertionError("should use SoundFile(fh)")),
        )
        with (
            watchdog(),
            mock.patch.object(wave, "open", side_effect=swap_to_fifo_during_probe),
            mock.patch.dict(sys.modules, {"soundfile": fake_sf}),
        ):
            pcm = wake_eval.load_pcm(clip)
        self.assertEqual(len(seen_file_obj), 1)
        self.assertTrue(hasattr(seen_file_obj[0], "fileno"))
        self.assertEqual(seen_bytes, [payload])
        samples = np.frombuffer(pcm, dtype="<i2")
        self.assertTrue(np.all(samples[:100] == 8192))

    def test_soundfile_stream_enforces_budget_blockwise(self) -> None:
        clip = self.root / "stream.flac"
        clip.write_bytes(b"fLaC" + bytes(64))
        reads = 0

        class LyingStreamSoundFile:
            def __init__(self, f):
                self.channels = 1
                self.samplerate = wake_eval.SAMPLE_RATE
                self.frames = 0  # understated header

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, frames=-1, dtype="float64", always_2d=True):
                nonlocal reads
                reads += 1
                return np.zeros((32, 1), dtype=np.float64)

        fake_sf = types.SimpleNamespace(SoundFile=LyingStreamSoundFile)
        # 32 frames * 2 bytes/int16 = 64 bytes per block; cap at 100 -> fails on 2nd block (128 > 100).
        with (
            mock.patch.object(wake_eval, "MAX_PCM_BYTES", 100),
            mock.patch.dict(sys.modules, {"soundfile": fake_sf}),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "decoded PCM exceeds 100 bytes"),
        ):
            wake_eval.load_pcm(clip)
        self.assertEqual(reads, 2)

    def test_coprime_sample_rate_refused_before_scipy_filter_allocation(self) -> None:
        # 16001 Hz is coprime with 16000 Hz -> reduced factor 16001 > MAX_RESAMPLE_FACTOR (4096).
        wav = write_wav_header_only(
            self.root / "prime_rate.wav",
            nchannels=1,
            sampwidth=2,
            framerate=16001,
            data_bytes=200,
            truncate_to_data=True,
        )
        fake_sig = types.SimpleNamespace(
            resample_poly=mock.Mock(side_effect=AssertionError("resample_poly called for huge factor"))
        )
        with (
            mock.patch.dict(
                sys.modules,
                {"scipy": types.SimpleNamespace(signal=fake_sig), "scipy.signal": fake_sig},
            ),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "unsupported resample ratio"),
        ):
            wake_eval.load_pcm(wav)

        # Also refused on the soundfile fallback path (FLAC with 16001 Hz).
        flac = self.root / "prime_rate.flac"
        flac.write_bytes(b"fLaC" + bytes(64))
        fake_sf = types.SimpleNamespace(
            read=lambda p, dtype="float64", always_2d=True: (np.zeros((10, 1)), 16001)
        )
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "soundfile": fake_sf,
                    "scipy": types.SimpleNamespace(signal=fake_sig),
                    "scipy.signal": fake_sig,
                },
            ),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "unsupported resample ratio"),
        ):
            wake_eval.load_pcm(flac)

    def test_resample_info_cap_refuses_huge_compressed_clip_before_read(self) -> None:
        flac = self.root / "huge.flac"
        flac.write_bytes(b"fLaC" + bytes(64))

        def info(path):
            return types.SimpleNamespace(
                frames=wake_eval.MAX_PCM_BYTES,
                channels=1,
                samplerate=wake_eval.SAMPLE_RATE,
            )

        def fail_read(*args, **kwargs):
            raise AssertionError("sf.read called past sf.info cap")

        fake_sf = types.SimpleNamespace(info=info, read=fail_read)
        with (
            mock.patch.dict(sys.modules, {"soundfile": fake_sf}),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "decoded PCM exceeds"),
        ):
            wake_eval.load_pcm(flac)

    def test_resample_upsample_ratio_cap_refuses_before_resample(self) -> None:
        flac = self.root / "lowrate.flac"
        flac.write_bytes(b"fLaC" + bytes(64))

        def read(path, dtype="float64", always_2d=True):
            # 100 samples at 80 Hz (factor 200 <= 4096) upsamples to 20,000 samples (40 KB).
            return np.zeros((100, 1), dtype=np.float64), 80

        fake_sf = types.SimpleNamespace(read=read)
        fake_sig = types.SimpleNamespace(
            resample_poly=mock.Mock(side_effect=AssertionError("resample_poly called past cap"))
        )
        with (
            mock.patch.object(wake_eval, "MAX_PCM_BYTES", 4096),
            mock.patch.dict(
                sys.modules,
                {"soundfile": fake_sf, "scipy": types.SimpleNamespace(signal=fake_sig), "scipy.signal": fake_sig},
            ),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "resampled audio exceeds"),
        ):
            wake_eval.load_pcm(flac)


class AtomicOutputWriteTests(TempDirCase):
    def test_planted_symlink_at_out_is_replaced_not_followed(self) -> None:
        victim = self.root / "victim.txt"
        victim.write_text("keep\n", encoding="utf-8")
        out = self.root / "results.json"
        out.symlink_to(victim)
        write_native_wav(self.root / "pos.wav")
        write_native_wav(self.root / "neg.wav")
        m = self.root / "m.json"
        m.write_text(json.dumps([
            {"file": "pos.wav", "label": "positive"},
            {"file": "neg.wav", "label": "negative"},
        ]), encoding="utf-8")

        rc, _stdout, stderr = run_cli([
            "--manifest", str(m),
            "--clips-dir", str(self.root),
            "--model", "unused",
            "--thresholds", "0.5",
            "--floors", "0",
            "--out", str(out),
        ])
        self.assertEqual(rc, 0, stderr)
        self.assertEqual(victim.read_text(encoding="utf-8"), "keep\n")
        self.assertFalse(out.is_symlink())
        self.assertTrue(out.is_file())
        self.assertEqual(len(json.loads(out.read_text(encoding="utf-8"))["rows"]), 1)

    def test_dangling_symlink_at_out_is_replaced_without_creating_target(self) -> None:
        target = self.root / "never-create.txt"
        out = self.root / "results.json"
        out.symlink_to(target)
        wake_eval._write_text_atomic(out, '{"ok": true}\n')
        self.assertFalse(target.exists())
        self.assertFalse(out.is_symlink())
        self.assertEqual(out.read_text(encoding="utf-8"), '{"ok": true}\n')

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
                "src": src, "dst": dst,
                "src_dir_fd": src_dir_fd, "dst_dir_fd": dst_dir_fd,
            })
            return real_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        with (
            mock.patch.object(wake_eval.os, "open", side_effect=spy_open),
            mock.patch.object(wake_eval.os, "replace", side_effect=spy_replace),
        ):
            wake_eval._write_text_atomic(out, "ok\n")
        self.assertEqual(len(seen_opens), 2)  # parent dir_fd, sibling temp file
        _tmp_path, tmp_flags, tmp_mode, tmp_dir_fd = seen_opens[1]
        expected = os.O_WRONLY | os.O_CREAT | os.O_EXCL | wake_eval._O_NOFOLLOW
        self.assertEqual(tmp_flags & expected, expected)
        self.assertEqual(tmp_mode, 0o600)
        self.assertIsNotNone(tmp_dir_fd)
        self.assertEqual(len(seen_replaces), 1)
        self.assertEqual(seen_replaces[0]["src_dir_fd"], tmp_dir_fd)
        self.assertEqual(seen_replaces[0]["dst_dir_fd"], tmp_dir_fd)
        self.assertEqual(os.stat(out).st_mode & 0o777, 0o600)
        # No leftover temp files remain after atomic replacement.
        self.assertEqual([p.name for p in self.root.iterdir()], [out.name])

    def test_replacing_private_0600_report_does_not_broaden_permissions(self) -> None:
        out = self.root / "private.json"
        out.write_text("old\n", encoding="utf-8")
        os.chmod(out, 0o600)
        wake_eval._write_text_atomic(out, '{"new": true}\n')
        self.assertEqual(os.stat(out).st_mode & 0o777, 0o600)

    def test_long_multibyte_output_name_writes_cleanly(self) -> None:
        # 240 UTF-8 bytes (80 3-byte chars): within NAME_MAX (255), would overflow
        # if the temp file copied 64+ chars of the output name.
        out = self.root / ("\u4e2d" * 80)
        wake_eval._write_text_atomic(out, '{"ok": true}\n')
        self.assertEqual(out.read_text(encoding="utf-8"), '{"ok": true}\n')

    def test_failed_write_leaves_existing_out_and_sibling_tmp_untouched(self) -> None:
        out = self.root / "results.json"
        out.write_text("previous\n", encoding="utf-8")
        other_tmp = self.root / ".wake_eval.keepme.tmp"
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
            mock.patch.object(wake_eval.os, "fdopen", side_effect=fake_fdopen),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "cannot write"),
        ):
            wake_eval._write_text_atomic(out, "new\n")

        self.assertEqual(out.read_text(encoding="utf-8"), "previous\n")
        self.assertEqual(other_tmp.read_text(encoding="utf-8"), "unrelated\n")
        # Only results.json and other_tmp remain; our own temp file was removed.
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), [other_tmp.name, out.name])

    def test_replace_into_directory_cleans_up_temp_file_and_reports_error(self) -> None:
        out_dir = self.root / "out_dir"
        out_dir.mkdir()
        with self.assertRaisesRegex(wake_eval.WakeEvalError, "cannot write"):
            wake_eval._write_text_atomic(out_dir, "data\n")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["out_dir"])

    def test_colliding_temp_name_retries_without_touching_existing_file(self) -> None:
        out = self.root / "results.json"
        planted = self.root / ".wake_eval.deadbeefdeadbeef.tmp"
        planted.write_text("do-not-overwrite\n", encoding="utf-8")
        tokens = iter(["deadbeefdeadbeef", "0123456789abcdef"])
        with mock.patch.object(wake_eval.secrets, "token_hex", side_effect=lambda n: next(tokens)):
            wake_eval._write_text_atomic(out, "fresh\n")
        self.assertEqual(planted.read_text(encoding="utf-8"), "do-not-overwrite\n")
        self.assertEqual(out.read_text(encoding="utf-8"), "fresh\n")

    def test_invalid_output_path_refused(self) -> None:
        for bad in (Path("."), Path(".."), Path("/"), Path("bad\x00.json")):
            with self.subTest(bad=bad), self.assertRaisesRegex(wake_eval.WakeEvalError, "invalid output path"):
                wake_eval._write_text_atomic(bad, "x\n")

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
                wake_eval.os, "fdopen",
                side_effect=lambda fd, *a, **kw: FailingCloseWriter(real_fdopen(fd, *a, **kw)),
            ),
            mock.patch.object(
                wake_eval.os, "replace",
                side_effect=AssertionError("os.replace called despite close error"),
            ),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "cannot write"),
        ):
            wake_eval._write_text_atomic(out, "new\n")
        self.assertEqual(out.read_text(encoding="utf-8"), "original\n")
        self.assertEqual([p.name for p in self.root.iterdir()], [out.name])

    def test_vfat_style_0644_temp_file_succeeds_with_zero_leftovers(self) -> None:
        # Non-POSIX mounts (e.g. vfat with fmask=0022) report 0644 on owned files.
        out = self.root / "results.json"
        real_fstat = os.fstat

        def fake_fmask_0022(fd):
            st = real_fstat(fd)
            return os.stat_result((
                (st.st_mode & ~0o777) | 0o644,
                st.st_ino, st.st_dev, st.st_nlink, st.st_uid, st.st_gid,
                st.st_size, st.st_atime, st.st_mtime, st.st_ctime,
            ))

        with mock.patch.object(wake_eval.os, "fstat", side_effect=fake_fmask_0022):
            wake_eval._write_text_atomic(out, '{"ok": true}\n')
        self.assertEqual(out.read_text(encoding="utf-8"), '{"ok": true}\n')
        self.assertEqual([p.name for p in self.root.iterdir()], [out.name])

    def test_group_writable_or_foreign_owned_temp_file_refused_before_writing(self) -> None:
        for label, uid_delta, mode in (
            ("group-writable-0664", 0, 0o664),
            ("foreign-owner", 1, 0o600),
        ):
            with self.subTest(label):
                sub = self.root / label
                sub.mkdir()
                out = sub / "results.json"
                out.write_text("original\n", encoding="utf-8")
                real_fstat = os.fstat

                def fake_bad_file(fd):
                    st = real_fstat(fd)
                    return os.stat_result((
                        (st.st_mode & ~0o777) | mode,
                        st.st_ino, st.st_dev, st.st_nlink, st.st_uid + uid_delta, st.st_gid,
                        st.st_size, st.st_atime, st.st_mtime, st.st_ctime,
                    ))

                with (
                    mock.patch.object(wake_eval.os, "fstat", side_effect=fake_bad_file),
                    mock.patch.object(
                        wake_eval.os, "fdopen",
                        side_effect=AssertionError("fdopen called on refused temp file"),
                    ),
                    self.assertRaisesRegex(wake_eval.WakeEvalError, "writable by others|not owned"),
                ):
                    wake_eval._write_text_atomic(out, "secret\n")
                self.assertEqual(out.read_text(encoding="utf-8"), "original\n")
                self.assertEqual([p.name for p in sub.iterdir()], [out.name])

    def test_group_writable_0775_parent_leaves_no_temp_files(self) -> None:
        sub = self.root / "umask002"
        sub.mkdir()
        os.chmod(sub, 0o775)
        out = sub / "results.json"
        wake_eval._write_text_atomic(out, '{"ok": true}\n')
        self.assertEqual(out.read_text(encoding="utf-8"), '{"ok": true}\n')
        self.assertEqual([p.name for p in sub.iterdir()], [out.name])

    def test_fdopen_failure_closes_fds_and_removes_temp_file(self) -> None:
        out = self.root / "results.json"
        closed: list[int] = []
        real_close = os.close

        def track_close(fd: int) -> None:
            closed.append(fd)
            real_close(fd)

        with (
            mock.patch.object(wake_eval.os, "close", side_effect=track_close),
            mock.patch.object(wake_eval.os, "fdopen", side_effect=RuntimeError("fdopen boom")),
            self.assertRaisesRegex(RuntimeError, "fdopen boom"),
        ):
            wake_eval._write_text_atomic(out, "data\n")
        # Both temp fd and parent dir_fd must be closed.
        self.assertEqual(len(closed), 2)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_exhausted_temp_collisions_fail_cleanly(self) -> None:
        out = self.root / "results.json"
        planted = self.root / ".wake_eval.aaaaaaaaaaaaaaaa.tmp"
        planted.write_text("existing\n", encoding="utf-8")
        with (
            mock.patch.object(wake_eval.secrets, "token_hex", return_value="a" * 16),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "cannot create unique temp output"),
        ):
            wake_eval._write_text_atomic(out, "data\n")
        self.assertEqual(planted.read_text(encoding="utf-8"), "existing\n")
        self.assertFalse(out.exists())

    def test_cli_handles_undefined_metrics_and_invalid_numeric_flags(self) -> None:
        clip = write_native_wav(self.root / "pos.wav")
        # Single-label manifest: fpr is None -> JSON null, stdout n/a.
        m = self.root / "single.json"
        m.write_text(json.dumps([{"file": clip.name, "label": "positive"}]), encoding="utf-8")
        out = self.root / "single_out.json"
        rc, stdout, stderr = run_cli([
            "--manifest", str(m),
            "--clips-dir", str(self.root),
            "--model", "unused",
            "--thresholds", "0.5",
            "--floors", "0",
            "--out", str(out),
        ])
        self.assertEqual(rc, 0, stderr)
        self.assertIn("n/a", stdout)
        self.assertIsNone(json.loads(out.read_text(encoding="utf-8"))["rows"][0]["fpr"])

        # Empty corpus list: wakes_per_hour is None -> JSON null, stdout n/a.
        empty_lst = self.root / "empty.txt"
        empty_lst.write_text("\n", encoding="utf-8")
        out2 = self.root / "empty_corpus.json"
        rc, stdout, stderr = run_cli([
            "--mode", "corpus",
            "--corpus-list", str(empty_lst),
            "--model", "unused",
            "--thresholds", "0.5",
            "--floors", "0",
            "--out", str(out2),
        ])
        self.assertEqual(rc, 0, stderr)
        self.assertIn("n/a", stdout)
        self.assertIsNone(json.loads(out2.read_text(encoding="utf-8"))["rows"][0]["wakes_per_hour"])

        # Invalid numeric flag reports concise wake_eval error and exits 1.
        rc, _stdout, stderr = run_cli([
            "--manifest", str(m),
            "--clips-dir", str(self.root),
            "--model", "unused",
            "--thresholds", "not-a-float",
            "--out", str(self.root / "bad_num.json"),
        ])
        self.assertEqual(rc, 1)
        self.assertIn("wake_eval: invalid numeric argument:", stderr)

    def test_cli_reports_missing_corpus_dir_and_bad_out_cleanly(self) -> None:
        out = self.root / "out.json"
        rc, _stdout, stderr = run_cli([
            "--mode", "corpus",
            "--corpus-dir", str(self.root / "no-such-dir"),
            "--model", "unused",
            "--out", str(out),
        ])
        self.assertEqual(rc, 1)
        self.assertIn("wake_eval:", stderr)
        self.assertIn("cannot read corpus directory", stderr)

        clip = write_native_wav(self.root / "a.wav")
        lst = self.root / "corpus.txt"
        lst.write_text(f"{clip}\n", encoding="utf-8")
        out_dir = self.root / "is_a_dir"
        out_dir.mkdir()
        rc, _stdout, stderr = run_cli([
            "--mode", "corpus",
            "--corpus-list", str(lst),
            "--model", "unused",
            "--thresholds", "0.5",
            "--floors", "0",
            "--out", str(out_dir),
        ])
        self.assertEqual(rc, 1)
        self.assertIn("wake_eval:", stderr)
        self.assertIn("cannot write", stderr)

    def test_low_rate_non_native_wav_upsample_cap_refused_before_resample(self) -> None:
        # 80 Hz WAV (factor 200 <= 4096) with 1000 frames is only 2044 bytes on disk,
        # but upsampling to 16 kHz yields 200,000 samples (400 KB).
        low = write_wav_header_only(
            self.root / "lowrate.wav",
            nchannels=1,
            sampwidth=2,
            framerate=80,
            data_bytes=2000,
            truncate_to_data=True,
        )
        with (
            mock.patch.object(wake_eval, "MAX_PCM_BYTES", 100_000),
            mock.patch.object(
                wake_eval,
                "_resample_stream",
                side_effect=AssertionError("resample called past upsample cap"),
            ),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "resampled audio exceeds"),
        ):
            wake_eval.load_pcm(low)

    def test_direct_resample_refuses_fifo(self) -> None:
        pipe = make_fifo(self.root / "direct.flac")
        fake_sf = types.SimpleNamespace(
            read=mock.Mock(side_effect=AssertionError("sf.read called on FIFO"))
        )
        with (
            watchdog(),
            mock.patch.dict(sys.modules, {"soundfile": fake_sf}),
            self.assertRaisesRegex(wake_eval.WakeEvalError, "non-regular"),
        ):
            wake_eval._resample_to_16k(pipe)


if __name__ == "__main__":
    unittest.main()
