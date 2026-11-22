"""Stage-boundary types for the offline voice runtime (V16/VX).

The assistant's offline loop is a chain of stages wired through a small
set of typed messages. Every later VX row implements against these
types; nothing else crosses a stage boundary:

    capture -> [PcmChunk] -> wake -> [WakeEvent] -> VAD endpointing
        -> [EndpointEvent] + [PcmSegment] -> STT -> [Transcript]
        -> LLM -> [TokenStream] -> sentence split -> [SentenceStream]
        -> TTS -> [PcmChunk] -> speaker

Conventions:

- All boundary types are frozen dataclasses. A stage may not mutate a
  message it received; downstream stages see exactly what upstream
  emitted.
- `seq` orders messages inside a session; `t` is the session clock in
  seconds (monotonic, capture epoch). Audio spans carry `t0`/`t1`.
- Streams (`TokenStream`, `SentenceStream`) are single-shot: their
  `parts` iterable is consumed once by the next stage. A stage that
  needs the text more than once must materialize it.
- `final=True` marks the last message of a logical unit (one decode, one
  reply, one playback run) so downstream stages flush without waiting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Protocol, TypeVar, runtime_checkable


@dataclass(frozen=True, slots=True)
class PcmChunk:
    """One block of PCM audio moving between capture, VAD, and playback."""

    pcm: bytes
    sample_rate: int
    channels: int = 1
    seq: int = 0
    t0: float = 0.0
    t1: float = 0.0
    final: bool = False

    @property
    def frame_count(self) -> int:
        return len(self.pcm) // (2 * self.channels)


@dataclass(frozen=True, slots=True)
class PcmSegment:
    """A complete, endpoint-bounded utterance handed to the STT stage."""

    pcm: bytes
    sample_rate: int
    channels: int = 1
    seq: int = 0
    t0: float = 0.0
    t1: float = 0.0

    @property
    def duration(self) -> float:
        return self.frame_count / self.sample_rate

    @property
    def frame_count(self) -> int:
        return len(self.pcm) // (2 * self.channels)


@dataclass(frozen=True, slots=True)
class WakeEvent:
    """The wake stage heard its word; VAD/recording should arm now."""

    model: str
    score: float
    t: float
    seq: int = 0


EndpointKind = Literal["speech_start", "speech_end"]


@dataclass(frozen=True, slots=True)
class EndpointEvent:
    """VAD boundary mark; pairs with the PcmSegment it brackets."""

    kind: EndpointKind
    t: float
    reason: str = ""
    seq: int = 0


@dataclass(frozen=True, slots=True)
class Transcript:
    """STT output for one PcmSegment; `final=False` is a partial."""

    text: str
    final: bool = True
    seq: int = 0


@dataclass(frozen=True, slots=True)
class TokenStream:
    """The LM stage's reply as it arrives; consumed once, in order."""

    parts: Iterable[str]
    seq: int = 0
    final: bool = True

    def __iter__(self) -> Iterator[str]:
        return iter(self.parts)


@dataclass(frozen=True, slots=True)
class SentenceStream:
    """Reply text re-framed as whole sentences for streaming TTS."""

    parts: Iterable[str]
    seq: int = 0
    final: bool = True

    def __iter__(self) -> Iterator[str]:
        return iter(self.parts)


InT = TypeVar("InT", contravariant=True)
OutT = TypeVar("OutT", covariant=True)


@runtime_checkable
class Stage(Protocol[InT, OutT]):
    """One pipeline stage: consumes `InT` messages, yields `OutT` messages.

    `push` answers zero or more output messages per input (a wake stage
    emits nothing until it fires; an endpointing stage emits a
    PcmSegment only when speech closes). `flush` drains whatever the
    stage was buffering at end-of-input — the sentence splitter's last
    partial sentence, the segmenter's trailing segment.
    """

    def push(self, item: InT) -> Iterable[OutT]:
        ...

    def flush(self) -> Iterable[OutT]:
        ...
