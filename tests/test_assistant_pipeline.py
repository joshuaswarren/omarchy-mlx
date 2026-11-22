"""Unit tests for the V16 stage-boundary interface (VX-01).

Every later VX row implements pipeline stages against these types. The
tests pin the contract: each boundary type survives a trip through a
Stage-shaped passthrough with its fields intact, and the Stage protocol
itself accepts a conforming class and rejects a non-conforming one.
"""

import sys
import unittest
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serve"))

from mlx_omarchy_assistant.pipeline import (  # noqa: E402
    EndpointEvent,
    PcmChunk,
    PcmSegment,
    SentenceStream,
    Stage,
    TokenStream,
    Transcript,
    WakeEvent,
)


class PassthroughStage:
    """A Stage that emits every message it is pushed, in order."""

    def __init__(self) -> None:
        self.buffer: list = []
        self.flushed = False

    def push(self, item) -> Iterable:
        self.buffer.append(item)
        return [item]

    def flush(self) -> Iterable:
        self.flushed = True
        return list(self.buffer)


class SilentStage:
    """Not a Stage: push exists but flush is missing."""

    def push(self, item) -> Iterable:
        return []


def round_trip(msg):
    stage = PassthroughStage()
    out = list(stage.push(msg))
    assert len(out) == 1
    assert out[0] is msg  # frozen messages pass through unmodified
    return out[0]


class TestStageProtocol(unittest.TestCase):
    def test_passthrough_is_a_stage(self):
        self.assertIsInstance(PassthroughStage(), Stage)

    def test_missing_flush_is_not_a_stage(self):
        self.assertNotIsInstance(SilentStage(), Stage)

    def test_push_and_flush_through_a_stage(self):
        stage = PassthroughStage()
        chunk = PcmChunk(pcm=b"\x01\x02" * 1280, sample_rate=16000, seq=1)
        list(stage.push(chunk))
        out = list(stage.flush())
        self.assertEqual(out, [chunk])
        self.assertTrue(stage.flushed)


class TestBoundaryTypes(unittest.TestCase):
    def test_pcm_chunk(self):
        chunk = PcmChunk(
            pcm=b"\x01\x02" * 1280,
            sample_rate=16000,
            seq=7,
            t0=1.0,
            t1=1.08,
            final=True,
        )
        got = round_trip(chunk)
        self.assertEqual(got.pcm, b"\x01\x02" * 1280)
        self.assertEqual(got.sample_rate, 16000)
        self.assertEqual(got.channels, 1)
        self.assertEqual(got.seq, 7)
        self.assertTrue(got.final)
        self.assertEqual(got.frame_count, 1280)

    def test_pcm_segment(self):
        seg = PcmSegment(
            pcm=b"\x00\x01" * 16000, sample_rate=16000, seq=2, t0=0.5, t1=1.5
        )
        got = round_trip(seg)
        self.assertEqual(got.frame_count, 16000)
        self.assertAlmostEqual(got.duration, 1.0)
        self.assertEqual((got.t0, got.t1), (0.5, 1.5))

    def test_wake_event(self):
        ev = WakeEvent(model="hey_jarvis", score=0.87, t=12.34, seq=1)
        got = round_trip(ev)
        self.assertEqual((got.model, got.score, got.t, got.seq), ("hey_jarvis", 0.87, 12.34, 1))

    def test_endpoint_event(self):
        ev = EndpointEvent(kind="speech_end", t=3.25, reason="silence", seq=9)
        got = round_trip(ev)
        self.assertEqual((got.kind, got.t, got.reason), ("speech_end", 3.25, "silence"))

    def test_transcript(self):
        tr = Transcript(text="turn the lights on", final=True, seq=3)
        got = round_trip(tr)
        self.assertEqual((got.text, got.final, got.seq), ("turn the lights on", True, 3))

    def test_token_stream(self):
        ts = TokenStream(parts=iter(["Hello", ",", " world", "!"]), seq=4)
        got = round_trip(ts)
        self.assertEqual(list(got), ["Hello", ",", " world", "!"])
        self.assertTrue(got.final)

    def test_sentence_stream(self):
        ss = SentenceStream(parts=iter(["Turn on the lights.", "Done."]), seq=5, final=True)
        got = round_trip(ss)
        self.assertEqual(list(got), ["Turn on the lights.", "Done."])

    def test_frozen(self):
        chunk = PcmChunk(pcm=b"\x00", sample_rate=16000)
        with self.assertRaises(Exception):
            chunk.pcm = b"\x01"  # frozen dataclass refuses mutation



if __name__ == "__main__":
    unittest.main()
