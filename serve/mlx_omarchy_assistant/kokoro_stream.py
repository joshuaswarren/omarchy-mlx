"""Streamed Kokoro decoding: audio leaves the worker while the vocoder runs.
…
the left before its last stage, so window audio sample s is utterance
sample 5 * f0 + s, f0 = 0 for the first window else 120 * c0 + 1.

Utterance cuts are FRAME-level: the pipeline front half (bert, duration
predictor, alignment, text encoder) runs ONCE over the chunk's whole
phoneme string, so predicted durations, F0 and N are exactly the
whole-call prediction, and each utterance is a frame range of that shared
timeline. Utterances decode with aligned-frame context on both sides and
join through a linear-gain crossfade; the seam renders share the same
features, so equal-power gains would add a spurious +3 dB bump.
stats=None keeps upstream's whole-call decoding
(the MLX_OMARCHY_KOKORO_STREAM=0 kill switch).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

SEGMENT_PHONEMES = 29
WINDOW_FRAMES = 24
CONTEXT_FRAMES = 1
UTTERANCE_CONTEXT_FRAMES = 8
SEAM_FADE_SAMPLES = 240
FRAME_SAMPLES = 600
STATS_FILE = Path(__file__).with_name("kokoro_gen_stats.npz")


def streamer_binding() -> dict:
    """Hashes binding a qualification receipt to this streamer build: the
    module itself and the frozen statistics file it loads."""
    import hashlib
    return {
        "kokoro_stream_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "kokoro_gen_stats_sha256": hashlib.sha256(
            STATS_FILE.read_bytes()).hexdigest(),
    }


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


def segment_boundaries(ps: str, budget: int = SEGMENT_PHONEMES) -> list[int]:
    """ps character positions after which to cut, choosing word boundaries
    of about `budget` phonemes, preferring a cut after punctuation once a
    segment is past 60% of the budget."""
    cuts: list[int] = []
    pos = 0
    size = 0
    for word in ps.split(" "):
        if word:
            pos += len(word)
            size += len(word)
            if size >= budget or (size >= 0.6 * budget and word[-1] in ",;:.!?"):
                cuts.append(pos)
                size = 0
        pos += 1
    if cuts:
        tail = ps[cuts[-1] + 1:].split()
        if sum(map(len, tail)) < budget / 3:
            cuts.pop()
    return cuts


def phoneme_segments(ps: str, budget: int = SEGMENT_PHONEMES) -> list[str]:
    """Cut a phoneme string at word boundaries into utterances of about
    `budget` phonemes; a tail under a third of the budget joins the
    segment before it."""
    cuts = [0] + segment_boundaries(ps, budget) + [len(ps)]
    return [ps[a:b].strip() for a, b in zip(cuts, cuts[1:]) if ps[a:b].strip()]


def cut_frames(ps: str, vocab: dict, pred_dur, cuts: list[int]) -> list[int]:
    """Absolute aligned-frame position after each ps cut position.

    The model's input tokens are the ps characters present in the vocab
    (in order) between two pad tokens, and pred_dur carries one duration
    per input token, so a cut after ps character c lands on the frame
    boundary that closes the vocab characters up to c."""
    import numpy as np
    cum = np.cumsum(np.asarray(pred_dur))
    frames = []
    for cut in cuts:
        n = sum(1 for ch in ps[:cut] if ch in vocab)
        frames.append(int(cum[1 + n - 1]))
    return frames


def utterance_ranges(cut_positions: list[int], frames: int) -> list[tuple[int, int]]:
    """[(first, end)] aligned-frame ranges covering [0, frames), one per
    utterance, in order."""
    edges = [f for f in cut_positions if 0 < f < frames]
    edges = sorted(set(edges))
    ranges = []
    first = 0
    for end in edges:
        if end > first:
            ranges.append((first, end))
            first = end
    if first < frames:
        ranges.append((first, frames))
    return ranges or [(0, frames)]


class _DecoderInputs:
    """Stand-in decoder: KokoroModel.__call__ hands it the decoder inputs."""

    def __init__(self):
        self.inputs = None

    def __call__(self, asr, F0, N, s):
        import mlx.core as mx
        self.inputs = (asr, F0, N, s)
        return mx.zeros((1, 2))


def sentence_inputs(pipe, text: str, voice: str) -> Iterator[tuple]:
    """(asr, F0, N, s, ranges) per pipeline chunk, in order.

    The pipeline's own English G2P, chunking and front half run ONCE over
    each chunk's whole phoneme string; `ranges` are the utterance frame
    ranges cut from the shared predicted timeline."""
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
                if not ps:
                    continue
                cuts = segment_boundaries(ps[:510])
                capture.inputs = None
                out = model(ps[:510], pack[len(ps[:510]) - 1], 1,
                            return_output=True)
                if capture.inputs is None or out.pred_dur is None:
                    continue
                frames = int(capture.inputs[0].shape[2])
                positions = cut_frames(ps[:510], model.vocab, out.pred_dur, cuts)
                yield (*capture.inputs, utterance_ranges(positions, frames))
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
        for asr, F0, N, s, ranges in sentence_inputs(self.pipe, text, voice):
            yield from self._stream_ranges(asr, F0, N, s, voice, ranges)

    def _stream_ranges(self, asr, F0_curve, N_curve, s, voice: str,
                       ranges: list[tuple[int, int]]) -> Iterator:
        """Decode each utterance range with aligned-frame context and join
        neighbours through a linear-gain crossfade."""
        import mlx.core as mx
        import numpy as np
        frames = int(asr.shape[2])
        fade = min(SEAM_FADE_SAMPLES, UTTERANCE_CONTEXT_FRAMES * FRAME_SAMPLES)
        pending = np.zeros(0, dtype=np.float32)
        pending_start = 0

        # The harmonic source and its STFT are built ONCE over the whole
        # chunk so every utterance slice inherits the absolute source phase
        # (a per-slice rebuild resets the phase accumulator and decorrelates
        # the slice audio from the whole call).
        gen = self.decoder.generator
        f0_up = gen.f0_upsamp(F0_curve[:, None].transpose(0, 2, 1))
        source, _, _ = gen.m_source(f0_up)
        source = mx.squeeze(source.transpose(0, 2, 1), axis=1)
        mag, phase = gen.stft.transform(source)
        har = mx.concatenate([mag, phase], axis=1).swapaxes(2, 1)
        mx.eval(har)

        def emit_upto(pos: int):
            nonlocal pending, pending_start
            pos = min(pos, pending_start + pending.size)
            if pos > pending_start:
                piece, pending = (pending[:pos - pending_start],
                                  pending[pos - pending_start:])
                pending_start = pos
                if piece.size:
                    yield np.ascontiguousarray(piece)

        for k, (first, end) in enumerate(ranges):
            ctx0 = max(0, first - UTTERANCE_CONTEXT_FRAMES)
            ctx1 = min(frames, end + UTTERANCE_CONTEXT_FRAMES)
            # F0/N are predicted at twice the aligned-frame rate
            # (the decoder's F0_conv/N_conv stride 2 halves them back).
            f0a, f0b = 2 * ctx0, 2 * ctx1
            start = ctx0 * FRAME_SAMPLES
            hold = fade // 2 if k < len(ranges) - 1 else 0
            cut_to = min(end * FRAME_SAMPLES - hold,
                         start + (ctx1 - ctx0) * FRAME_SAMPLES)
            faded = k == 0
            for piece in self._render_pieces(
                    asr[:, :, ctx0:ctx1],
                    F0_curve[:, f0a:f0b] if F0_curve.ndim == 2
                    else F0_curve[:, :, f0a:f0b],
                    N_curve[:, f0a:f0b] if N_curve.ndim == 2
                    else N_curve[:, :, f0a:f0b],
                    s, voice, har, ctx0):
                if k > 0 and not faded:
                    cut = first * FRAME_SAMPLES
                    half = fade // 2
                    b0 = cut - half - pending_start
                    c0 = cut - half - start
                    width = min(fade, pending.size - b0, piece.size - c0)
                    if width > 0:
                        # Linear gain ramp: the two renders derive from the
                        # SAME whole-sentence features, so equal-power gains
                        # would add a spurious +3 dB bump.
                        t = (np.arange(width) + 0.5) / width
                        a, b = pending[b0:b0 + width], piece[c0:c0 + width]
                        pending = np.concatenate([pending[:b0], a + t * (b - a)])
                        faded = True
                        piece = piece[c0 + width:]
                pending = np.concatenate([pending, piece])
                yield from emit_upto(cut_to)
        yield from emit_upto(pending_start + pending.size)

    def _render_pieces(self, asr, F0_curve, N_curve, s, voice: str,
                       har=None, ctx0: int = 0):
        """Yield float32 window pieces over one context slice: the unchanged
        decoder for one-window utterances, else windowed last-stage decoding.

        `har` is the whole-chunk harmonic STFT and `ctx0` the slice's first
        aligned frame, so every window reads the source at its ABSOLUTE
        position (phase-continuous with the whole call)."""
        import mlx.core as mx
        import numpy as np
        if int(asr.shape[2]) <= WINDOW_FRAMES:
            yield self._whole(asr, F0_curve, N_curve, s)
            return
        self._table["current"] = self._stats[voice]
        try:
            yield from self._decode(asr, F0_curve, N_curve, s, har, ctx0)
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

    def _decode(self, asr, F0_curve, N_curve, s, har_abs=None, ctx0: int = 0,
                keep=None) -> Iterator:
        import mlx.core as mx
        import numpy as np
        gen = self.decoder.generator
        f0_up = gen.f0_upsamp(F0_curve[:, None].transpose(0, 2, 1))
        source, _, _ = gen.m_source(f0_up)
        source = mx.squeeze(source.transpose(0, 2, 1), axis=1)
        mag, phase = gen.stft.transform(source)
        har_local = mx.concatenate([mag, phase], axis=1).swapaxes(2, 1)
        if har_abs is not None:
            # Absolute-position harmonic STFT: the slice reads the source at
            # its whole-call position (phase-continuous); the locally built
            # graph is used only for its lazy shape and never evaluated.
            har_use = har_abs[:, 120 * ctx0:120 * ctx0 + har_local.shape[1]]
            import json as _json
            print(_json.dumps({"event": "har_shapes", "frames": int(asr.shape[2]),
                               "ctx0": ctx0,
                               "har_abs": list(har_abs.shape),
                               "har_local": list(har_local.shape)}), flush=True)
        else:
            har_use = har_local
        x = self._first_stage(self._low_stack(asr, F0_curve, N_curve, s), s,
                              har_use)
        mx.eval(har_use, x)
        frames = int(asr.shape[2])
        p0, end = keep or (0, frames)
        while p0 < end:
            p1 = min(end, p0 + WINDOW_FRAMES)
            c0 = max(0, p0 - CONTEXT_FRAMES)
            c1 = min(frames, p1 + CONTEXT_FRAMES)
            audio, f0 = self._last_stage(x, s, har_use, c0, c1)
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
