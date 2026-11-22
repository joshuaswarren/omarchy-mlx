"""Red tests for the silero VAD stage in the voice loop.

The VAD replaces the energy gate as the speech decision when opted in
(--vad silero); the energy gate survives only as an optional cheap
pre-filter that skips the model for silent chunks. The ONNX session is
stubbed, so CI stays CPU-fast with no model download; the real-model
mini50 table is the receipt, not CI.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import vad  # noqa: E402
from mlx_omarchy_assistant.voice_loop import Endpointer  # noqa: E402

SAMPLE_RATE = 16_000


class StubSession:
    """ONNX stand-in: returns a scripted probability per frame."""

    def __init__(self, probs, threshold=0.5):
        self.probs = list(probs)
        self.calls = []  # (input samples, state snapshot flag)

    def run(self, _out_names, feed):
        x = feed["input"]
        self.calls.append(int(x.shape[1]))
        prob = self.probs.pop(0) if self.probs else 0.0
        return (np.array([[prob]], dtype=np.float32),
                np.zeros((2, 1, 128), dtype=np.float32))


def tone(seconds, freq=0.3, amp=8000):
    return (np.sin(np.arange(int(seconds * SAMPLE_RATE)) * freq) * amp).astype("int16").tobytes()


def mic_chunk(seconds=0.04, freq=0.3, amp=8000):
    """One voice_loop mic step: 640 samples = 1280 bytes."""
    return tone(seconds, freq, amp)


class StubWake:
    """Wake stand-in: the first chunk opens the turn, the rest are quiet."""

    def __init__(self):
        self.calls = 0

    def process(self, pcm):
        self.calls += 1
        return self.calls == 1


class SileroVadTests(unittest.TestCase):
    def test_frames_run_at_512_samples_with_state(self):
        session = StubSession([0.1, 0.9])
        model = vad.SileroVad("unused", session=session)
        # Two mic chunks (1280 samples) buffer into two 512-sample frames.
        self.assertFalse(model.process(mic_chunk()))
        self.assertTrue(model.process(mic_chunk()))
        self.assertEqual(session.calls, [576, 576])

    def test_threshold_crossing(self):
        session = StubSession([0.49, 0.51])
        model = vad.SileroVad("unused", session=session, threshold=0.5)
        self.assertFalse(model.process(mic_chunk()))
        self.assertTrue(model.process(mic_chunk()))

    def test_partial_frames_wait_for_full_data(self):
        session = StubSession([0.9, 0.9])
        model = vad.SileroVad("unused", session=session)
        # One mic chunk = 640 samples: one frame runs, 128 samples held.
        model.process(mic_chunk())
        self.assertEqual(session.calls, [576])
        model.process(mic_chunk())  # the remainder completes frame two
        self.assertEqual(len(session.calls), 2)

    def test_reset_clears_buffer_and_state(self):
        session = StubSession([0.9, 0.9])
        model = vad.SileroVad("unused", session=session)
        model.process(mic_chunk())
        model.reset()
        model.process(mic_chunk())
        # reset dropped the 128-sample remainder: frame two is new audio only
        self.assertEqual(len(session.calls), 2)

    def test_frames_are_normalized_to_unit_scale(self):
        # silero v5 expects float audio in [-1, 1]; raw int16 values (up to
        # 32767) would saturate the model's mel input.
        class RecordingSession(StubSession):
            def __init__(self):
                super().__init__([0.5])
                self.frames = []

            def run(self, _names, feed):
                frame = feed["input"]
                self.frames.append(frame)
                return (np.array([[0.5]], dtype=np.float32),
                        np.zeros((2, 1, 128), dtype=np.float32))

        session = RecordingSession()
        model = vad.SileroVad("unused", session=session)
        model.process(mic_chunk())  # sine at amplitude 8000
        self.assertEqual(session.frames[0].shape[1], 576)  # 64 ctx + 512
        peak = float(np.max(np.abs(session.frames[0])))
        self.assertGreater(peak, 0.0)
        self.assertLessEqual(peak, 1.0)

    def test_context_carries_the_previous_tail_without_rescaling(self):
        # upstream's OnnxWrapper feeds concat(context, frame) where the
        # context is the previous input's last 64 samples, already in
        # [-1, 1]. Dividing the carried context again crushes it (0.5 ->
        # 1.5e-5) and blinds the model.
        class RecordingSession(StubSession):
            def __init__(self):
                super().__init__([0.5, 0.5])
                self.frames = []

            def run(self, _names, feed):
                frame = feed["input"]
                self.frames.append(frame)
                return (np.array([[0.5]], dtype=np.float32),
                        np.zeros((2, 1, 128), dtype=np.float32))

        session = RecordingSession()
        model = vad.SileroVad("unused", session=session)
        model.process(mic_chunk())
        model.process(mic_chunk())
        first, second = session.frames[0], session.frames[1]
        np.testing.assert_allclose(
            second[0, :64], first[0, -64:], rtol=0, atol=1e-9)


class PreparePinTests(unittest.TestCase):
    DATA = b"x" * 64
    DIGEST = None  # filled in setUp: the sha256 of DATA

    def setUp(self):
        import hashlib

        type(self).DIGEST = hashlib.sha256(self.DATA).hexdigest()

    def test_wrong_bytes_refused(self):
        with self.assertRaises(vad.VadModelError):
            vad.prepare(
                Path(tempfile.mkdtemp(prefix="vad-prepare-bad-")),
                fetch=lambda url: b"not the model", digest=self.DIGEST)

    def test_right_bytes_accepted_and_pinned_path(self):
        path = vad.prepare(
            Path(tempfile.mkdtemp(prefix="vad-prepare-ok-")),
            fetch=lambda url: self.DATA, digest=self.DIGEST)
        self.assertEqual(path.read_bytes(), self.DATA)
        self.assertIn("vad", str(path))


class PinAgreementTests(unittest.TestCase):
    """vad.py and install.sh must carry the same model pin: the installer
    stages what the runtime verifies, or the default path silently breaks."""

    def test_installer_pin_matches_vad_module(self):
        commit = vad.MODEL_URL.split("/silero-vad/")[1].split("/")[0]
        install = (REPO_ROOT / "install.sh").read_text()
        self.assertIn(commit, install)
        self.assertIn(vad.MODEL_SHA256, install)
        self.assertIn("silero_vad.onnx", install)


class EndpointerVadTests(unittest.TestCase):
    class FixedVad:
        def __init__(self, voiced):
            self.voiced_values = list(voiced)
            self.asked = 0
            self.resets = 0

        def voiced(self, pcm):
            self.asked += 1
            return self.voiced_values.pop(0)

        def reset(self):
            self.resets += 1

    def quiet_chunk(self):
        return b"\x00" * 1280  # RMS 0: below any energy floor

    def test_vad_replaces_the_energy_decision(self):
        # Silence by energy, speech by VAD: the VAD decision wins, so quiet
        # speech still holds the turn open (the energy gate's blind spot).
        vad_stub = self.FixedVad([True, False, False, False, False, False])
        ep = Endpointer(StubWake(), rms_floor=300.0, end_silence_s=0.1,
                        vad=vad_stub)
        ep.feed(self.quiet_chunk())  # wake opens the turn (not the VAD)
        ep.feed(self.quiet_chunk())  # VAD-voiced: the buffer opens
        resets_before_close = vad_stub.resets  # __init__ already reset once
        out = None
        for _ in range(3):
            out = ep.feed(self.quiet_chunk())  # VAD unvoiced -> closes
            if out is not None:
                break
        self.assertIsNotNone(out)
        # a finished turn must reset the VAD: RNN state and buffered PCM
        # from one turn must not leak into the next
        self.assertEqual(vad_stub.resets, resets_before_close + 1)

    def test_finished_turn_zeroes_the_vad_state(self):
        # a reset that only counts calls would pass a no-op; this one proves
        # the model's state really is zeroed for the next turn
        class StatefulSession(StubSession):
            def __init__(self):
                # 0.9 for the first (voiced) frame, 0.2 afterwards so the
                # quiet chunks close the turn
                super().__init__([0.9] + [0.2] * 50)
                self.states = []

            def run(self, _names, feed):
                self.states.append(np.array(feed["state"]))
                prob = self.probs.pop(0) if self.probs else 0.0
                return (np.array([[prob]], dtype=np.float32),
                        np.ones((2, 1, 128), dtype=np.float32))

        class AlwaysWake:
            def process(self, pcm):
                return True

        session = StatefulSession()
        model = vad.SileroVad("unused", session=session)
        ep = Endpointer(AlwaysWake(), rms_floor=300.0, end_silence_s=0.1,
                        vad=model)
        ep.feed(self.quiet_chunk())  # wake opens the turn
        ep.feed(tone(0.04))          # voiced: the buffer opens
        for _ in range(3):
            ep.feed(self.quiet_chunk())  # VAD unvoiced -> the turn closes
        self.assertTrue(session.states)  # the model ran
        self.assertTrue(any(np.any(s != 0.0) for s in session.states[:-1]))
        before = len(session.states)
        ep.feed(self.quiet_chunk())  # the next turn opens
        ep.feed(tone(0.04))          # the VAD runs again
        self.assertGreater(len(session.states), before)
        self.assertTrue(np.all(session.states[before] == 0.0))

    def test_prefilter_skips_the_model_for_silent_chunks(self):
        vad_stub = self.FixedVad([True])
        ep = Endpointer(StubWake(), rms_floor=300.0, vad=vad_stub,
                        prefilter_floor=300.0)
        ep.feed(self.quiet_chunk())  # wake opens the turn
        ep.feed(self.quiet_chunk())  # RMS 0: prefilter rejects, VAD not asked
        self.assertEqual(vad_stub.asked, 0)
        ep.feed(tone(0.04))  # loud: the VAD decides
        self.assertEqual(vad_stub.asked, 1)

    def test_default_stays_the_energy_gate(self):
        # No vad passed: a silent chunk (RMS 0) sits below the shipped
        # floor, never opens the turn buffer, and produces no utterance
        ep = Endpointer(StubWake(), rms_floor=300.0)
        ep.feed(self.quiet_chunk())  # wake opens the turn
        for _ in range(3):
            self.assertIsNone(ep.feed(self.quiet_chunk()))
        self.assertEqual(ep._buf, [])


_TMP = tempfile.TemporaryDirectory(prefix="vad-cli-home-")
MIN_HOME = _TMP.name
MIN_ARGS = ["--home", MIN_HOME, "--reply-url", "http://127.0.0.1:9"]


class DefaultVadTests(unittest.TestCase):
    """The voice loop defaults to silero once the installer prepared the
    model; an absent model falls back to the energy gate with one notice."""

    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="vad-home-"))
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)

    def test_parser_defaults_to_unset_silero(self):
        from mlx_omarchy_assistant.voice_loop import build_arg_parser

        # unset = silero if the installer prepared it, else the energy
        # fallback; --vad silero explicitly allows the download
        args = build_arg_parser().parse_args([*MIN_ARGS])
        self.assertIsNone(args.vad)
        self.assertEqual(args.vad_threshold, 0.5)
        self.assertEqual(
            build_arg_parser().parse_args([*MIN_ARGS, "--vad", "energy"]).vad,
            "energy")
        self.assertEqual(
            build_arg_parser().parse_args([*MIN_ARGS, "--vad", "silero"]).vad,
            "silero")

    def test_prepared_model_builds_the_silero_endpointer(self):
        from unittest.mock import patch

        from mlx_omarchy_assistant.voice_loop import (
            build_arg_parser, build_endpointer)

        args = build_arg_parser().parse_args([*MIN_ARGS])
        sentinel = self.home / "vad" / "silero_vad.onnx"
        with patch("mlx_omarchy_assistant.vad.prepared", return_value=sentinel):
            endpointer, notice = build_endpointer(
                args, self.home, session=StubSession([0.1]))
        self.assertIsNone(notice)
        from mlx_omarchy_assistant.vad import SileroVad
        self.assertIsInstance(endpointer._vad, SileroVad)

    def test_corrupt_model_file_falls_back(self):
        # a staged file whose bytes do not match the pin is not usable
        home = Path(tempfile.mkdtemp(prefix="vad-corrupt-"))
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        (home / "vad").mkdir(parents=True)
        (home / "vad" / "silero_vad.onnx").write_bytes(b"not the model")
        from mlx_omarchy_assistant.voice_loop import (
            build_arg_parser, build_endpointer)

        args = build_arg_parser().parse_args([*MIN_ARGS])
        endpointer, notice = build_endpointer(args, home)
        self.assertIsNone(endpointer._vad)
        self.assertIn("silero", notice)
        self.assertIn("energy", notice)

    def test_absent_model_falls_back_with_one_notice(self):
        from mlx_omarchy_assistant.voice_loop import (
            build_arg_parser, build_endpointer)

        args = build_arg_parser().parse_args([*MIN_ARGS])
        endpointer, notice = build_endpointer(args, self.home)
        self.assertIsNone(endpointer._vad)
        self.assertIn("silero", notice)
        self.assertIn("energy", notice)


if __name__ == "__main__":
    unittest.main()
