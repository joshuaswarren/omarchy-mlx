"""Streamed Kokoro decoding: audio leaves the worker while the vocoder runs.

KokoroPipeline returns no audio until the whole decoder has run over a
whole phoneme chunk. This module keeps mlx_audio 0.5.6's own G2P, chunking,
front half and decoder layers, and changes only the schedule:

1. Each phoneme chunk is cut at word boundaries into utterances of about
   SEGMENT_PHONEMES phonemes (~2 s of audio), so no utterance's up-front
   work delays the first audio or outruns the audio already queued. The
   silence the model predicts for an utterance's boundary pad tokens is
   trimmed to PAD_KEEP_FRAMES at each such cut, so a cut adds no pause.
2. Per utterance, the low-rate decoder stack and the generator's first
   upsampling stage run over the whole utterance. Stage 0 carries almost
   all of the per-utterance AdaIN statistics error a window would make
   (2026-10-03: exact stage-0 statistics lift corr vs the whole call from
   0.971 to 0.994, the run-to-run floor is 0.990), so it stays exact.
3. The generator's last stage (~61% of decoder cost) runs in windows of
   WINDOW_FRAMES aligned frames with CONTEXT_FRAMES of context per side,
   trimmed; its AdaIN layers use frozen per-voice statistics calibrated on
   a fixed corpus (scripts/kokoro_gen_stats.py, shipped in STATS_FILE).
   Each window's audio is yielded as soon as it is computed.

An utterance that fits in one window runs the whole decoder unchanged
(bit-identical to the whole call at the same cost), and
MLX_OMARCHY_KOKORO_STREAM=0 turns the schedule off entirely: the pipeline
then decodes each phoneme chunk in one whole call, as upstream does.

Alignment (measured shapes): one aligned frame = 600 samples = 20 stage-0
frames = 120 harmonic frames; the generator reflection-pads one frame on
the left before its last stage, so window audio sample s is utterance
sample 5 * f0 + s, f0 = 0 for the first window else 120 * c0 + 1.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

SEGMENT_PHONEMES = 29
PAD_KEEP_FRAMES = 1
WINDOW_FRAMES = 24
CONTEXT_FRAMES = 1
FRAME_SAMPLES = 600
STATS_FILE = Path(__file__).with_name("kokoro_gen_stats.npz")


def last_stage_adains(decoder) -> dict:
    """{name: AdaIN1d} for the AdaIN layers of the generator's last
    upsampling stage (its noise block and its resblocks)."""
    import mlx.nn as nn
    from mlx_audio.tts.models.kokoro.istftnet import AdaIN1d
    gen = decoder.generator
    stage = gen.num_upsamples - 1
    roots = {f"generator.noise_res[{stage}]": gen.noise_res[stage]}
    for j in range(gen.num_kernels):
        k = stage * gen.num_kernels + j
        roots[f"generator.resblocks[{k}]"] = gen.resblocks[k]
    found = {}

    def walk(mod, prefix):
        if isinstance(mod, AdaIN1d):
            found[prefix] = mod
        if isinstance(mod, nn.Module):
            for key, value in mod.items():
                walk(value, f"{prefix}.{key}")
        elif isinstance(mod, (list, tuple)):
            for i, value in enumerate(mod):
                walk(value, f"{prefix}[{i}]")

    for prefix, mod in roots.items():
        walk(mod, prefix)
    return found


def load_stats(path: Path, voices, revision: str) -> dict:
    """{voice: {layer: (mean, var)}} from STATS_FILE, checked against the
    pinned pack revision and the expected voice set."""
    import numpy as np
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("pack_revision") != revision:
            raise ValueError(
                f"{path.name} was calibrated for pack revision "
                f"{meta.get('pack_revision')}, not {revision}")
        stats: dict = {voice: {} for voice in voices}
        for name in data.files:
            if name == "meta":
                continue
            voice, layer, kind = name.split("|")
            if voice in stats:
                stats[voice].setdefault(layer, {})[kind] = data[name]
    missing = [v for v, layers in stats.items() if not layers]
    if missing:
        raise ValueError(f"{path.name} has no statistics for {missing}")
    return {voice: {layer: (e["mean"], e["var"]) for layer, e in layers.items()}
            for voice, layers in stats.items()}


def phoneme_segments(ps: str, budget: int = SEGMENT_PHONEMES) -> list[str]:
    """Cut a phoneme string at word boundaries into utterances of about
    `budget` phonemes, preferring a cut after punctuation once a segment
    is past 60% of the budget; a tail under a third of the budget joins
    the segment before it."""
    words = ps.split()
    segments: list[list[str]] = [[]]
    size = 0
    for word in words:
        segments[-1].append(word)
        size += len(word)
        if size >= budget or (size >= 0.6 * budget and word[-1] in ",;:.!?"):
            segments.append([])
            size = 0
    if not segments[-1]:
        segments.pop()
    if len(segments) > 1 and sum(map(len, segments[-1])) < budget / 3:
        segments[-2].extend(segments.pop())
    return [" ".join(seg) for seg in segments]


class _DecoderInputs:
    """Stand-in decoder: KokoroModel.__call__ hands it the decoder inputs."""

    def __init__(self):
        self.inputs = None

    def __call__(self, asr, F0, N, s):
        import mlx.core as mx
        self.inputs = (asr, F0, N, s)
        return mx.zeros((1, 2))


def segment_inputs(pipe, text: str, voice: str) -> Iterator[tuple]:
    """(asr, F0, N, s, keep) per utterance, in order, from the pipeline's
    own English G2P, chunking and front half; `keep` = (first, end) aligned
    frames to play."""
    pack = pipe.load_voice(voice)
    model = pipe.model
    decoder, capture = model.decoder, _DecoderInputs()
    model.decoder = capture
    try:
        for graphemes in re.split(r"\n+", text.strip()):
            if not graphemes.strip():
                continue
            _, tokens = pipe.g2p(graphemes)
            for _gs, ps, _tks in pipe.en_tokenize(tokens):
                segments = phoneme_segments(ps[:510]) if ps else []
                for k, segment in enumerate(segments):
                    capture.inputs = None
                    out = model(segment, pack[len(segment) - 1], 1, return_output=True)
                    if capture.inputs is None:
                        continue
                    frames = int(capture.inputs[0].shape[2])
                    lead, trail = int(out.pred_dur[0]), int(out.pred_dur[-1])
                    first = max(0, lead - PAD_KEEP_FRAMES) if k > 0 else 0
                    end = frames - (max(0, trail - PAD_KEEP_FRAMES) if k < len(segments) - 1 else 0)
                    yield (*capture.inputs, (first, max(first, end)))
    finally:
        model.decoder = decoder


def _frozen_norm_class():
    import mlx.core as mx
    import mlx.nn as nn

    class FrozenNorm(nn.Module):
        """InstanceNorm with statistics read from a shared per-voice table."""

        def __init__(self, table: dict, layer: str, inner):
            super().__init__()
            self._table = table
            self._layer = layer
            self._inner = inner

        def __call__(self, x):
            current = self._table["current"]
            if current is None:
                return self._inner(x)
            mean, var = current[self._layer]
            return (x - mean.astype(x.dtype)) / mx.sqrt(var.astype(x.dtype) + self._inner.eps)

    return FrozenNorm


class KokoroStreamer:
    """Streams a KokoroPipeline's audio window by window.

    Construction wraps the InstanceNorm inside each last-stage AdaIN layer
    so that, while a stream runs, it reads the selected voice's frozen
    statistics, and otherwise behaves exactly as before; the worker owns
    the model, so nothing else sees the wrap. stats=None keeps upstream's
    whole-call decoding (the MLX_OMARCHY_KOKORO_STREAM=0 kill switch).
    """

    def __init__(self, pipe, stats: dict | None):
        import mlx.core as mx
        self.pipe = pipe
        self.decoder = pipe.model.decoder
        self._table: dict = {"current": None}
        self._stats = None
        if stats is None:
            return
        layers = last_stage_adains(self.decoder)
        for voice, table in stats.items():
            if set(table) != set(layers):
                raise ValueError(
                    f"frozen statistics for {voice} do not match the generator's "
                    f"{len(layers)} last-stage AdaIN layers")
        self._stats = {voice: {layer: (mx.array(m)[None, :, None], mx.array(v)[None, :, None])
                               for layer, (m, v) in table.items()}
                       for voice, table in stats.items()}
        frozen = _frozen_norm_class()
        for name, adain in layers.items():
            adain.norm = frozen(self._table, name, adain.norm)

    def __call__(self, text: str, voice: str) -> Iterator:
        """Yield float32 numpy chunks of 24 kHz audio, in order."""
        import numpy as np
        if self._stats is None:
            for result in self.pipe(text, voice=voice, speed=1.0):
                audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                if audio.size:
                    yield audio
            return
        if voice not in self._stats:
            raise ValueError(f"no frozen generator statistics for voice {voice}")
        for asr, F0, N, s, keep in segment_inputs(self.pipe, text, voice):
            if int(asr.shape[2]) <= WINDOW_FRAMES:
                yield self._whole(asr, F0, N, s, keep)
                continue
            self._table["current"] = self._stats[voice]
            try:
                yield from self._decode(asr, F0, N, s, keep)
            finally:
                self._table["current"] = None

    def _whole(self, asr, F0_curve, N_curve, s, keep=None):
        """The unchanged decoder over one short utterance."""
        import mlx.core as mx
        import numpy as np
        audio = self.decoder(asr, F0_curve, N_curve, s)[0]
        mx.eval(audio)
        first, end = keep or (0, int(asr.shape[2]))
        audio = np.asarray(audio).reshape(-1)[FRAME_SAMPLES * first:FRAME_SAMPLES * end]
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _decode(self, asr, F0_curve, N_curve, s, keep=None) -> Iterator:
        import mlx.core as mx
        import numpy as np
        gen = self.decoder.generator
        f0_up = gen.f0_upsamp(F0_curve[:, None].transpose(0, 2, 1))
        source, _, _ = gen.m_source(f0_up)
        source = mx.squeeze(source.transpose(0, 2, 1), axis=1)
        mag, phase = gen.stft.transform(source)
        har = mx.concatenate([mag, phase], axis=1).swapaxes(2, 1)
        x = self._first_stage(self._low_stack(asr, F0_curve, N_curve, s), s, har)
        mx.eval(har, x)
        frames = int(asr.shape[2])
        p0, end = keep or (0, frames)
        while p0 < end:
            p1 = min(end, p0 + WINDOW_FRAMES)
            c0 = max(0, p0 - CONTEXT_FRAMES)
            c1 = min(frames, p1 + CONTEXT_FRAMES)
            audio, f0 = self._last_stage(x, s, har, c0, c1)
            piece = audio[FRAME_SAMPLES * p0 - 5 * f0:FRAME_SAMPLES * p1 - 5 * f0]
            if piece.size:
                yield np.ascontiguousarray(piece, dtype=np.float32)
            p0 = p1

    def _low_stack(self, asr, F0_curve, N_curve, s):
        """Decoder.__call__ up to the generator."""
        import mlx.core as mx
        dec = self.decoder
        F0 = dec.F0_conv(F0_curve[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        N = dec.N_conv(N_curve[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        x = dec.encode(mx.concatenate([asr, F0, N], axis=1), s)
        asr_res = dec.asr_res[0](asr.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        residual = True
        for block in dec.decode:
            if residual:
                x = mx.concatenate([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if getattr(block, "upsample_type", "none") != "none":
                residual = False
        return x

    def _first_stage(self, x, s, har):
        """Generator.__call__ stage 0 over the whole utterance (the model
        has exactly two upsampling stages)."""
        import mlx.core as mx
        gen = self.decoder.generator
        x = mx.where(x > 0, x, x * 0.1)
        source = gen.noise_res[0](gen.noise_convs[0](har).swapaxes(2, 1), s)
        x = gen.ups[0](x.swapaxes(2, 1), mx.conv_transpose1d).swapaxes(2, 1) + source
        total = None
        for j in range(gen.num_kernels):
            r = gen.resblocks[j](x, s)
            total = r if total is None else total + r
        return total / gen.num_kernels

    def _last_stage(self, x0, s, har, c0, c1):
        """Generator.__call__ stage 1 and the inverse STFT over aligned
        frames [c0, c1) of the stage-0 output."""
        import mlx.core as mx
        import numpy as np
        gen = self.decoder.generator
        x = x0[:, :, 20 * c0:20 * c1]
        f0 = 0 if c0 == 0 else 120 * c0 + 1
        x = mx.where(x > 0, x, x * 0.1)
        source = gen.noise_res[1](gen.noise_convs[1](har[:, f0:120 * c1 + 1, :]).swapaxes(2, 1), s)
        x = gen.ups[1](x.swapaxes(2, 1), mx.conv_transpose1d).swapaxes(2, 1)
        if c0 == 0:
            x = gen.reflection_pad(x)
        x = x + source
        total = None
        for j in range(gen.num_kernels):
            r = gen.resblocks[gen.num_kernels + j](x, s)
            total = r if total is None else total + r
        x = total / gen.num_kernels
        x = mx.where(x > 0, x, x * 0.01)
        x = gen.conv_post(x.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
        spec = mx.exp(x[:, :gen.post_n_fft // 2 + 1, :])
        phase = mx.sin(x[:, gen.post_n_fft // 2 + 1:, :])
        audio = gen.stft.inverse(spec, phase)
        mx.eval(audio)
        return np.asarray(audio).reshape(-1), f0
