"""Trace Parakeet TDT greedy decode steps (stock mlx CPU) for the failing upload and a working variant."""
import importlib.util
import sys
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

spec = importlib.util.find_spec("mlx_audio.stt.models")
sys.modules[spec.name] = importlib.util.module_from_spec(spec)
from mlx_audio.stt.utils import load_model  # noqa: E402
from mlx_audio.stt.models.parakeet import audio as pk_audio  # noqa: E402

w = wave.open(sys.argv[1])
x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
model = load_model(Path("/tmp/sig/parakeet-tdt-0.6b-v3"))
print("durations", model.durations, "max_symbols", model.max_symbols, "blank", model.blank_id)

for label, samples in (("fail", x), ("pad160", np.concatenate([x, np.zeros(160, np.float32)]))):
    mel = pk_audio.log_mel_spectrogram(mx.array(samples), model.preprocessor_config)
    feats, lengths = model.encoder(mel if mel.ndim == 3 else mx.expand_dims(mel, 0))
    mx.eval(feats, lengths)
    print(label, "mel", mel.shape, "feats", feats.shape, "lengths", lengths.tolist(),
          "mel finite", bool(mx.all(mx.isfinite(mel)).item()), "feat finite", bool(mx.all(mx.isfinite(feats)).item()))
    steps, t, last, new = [], 0, model.blank_id, 0
    hidden = model._make_initial_decoder_state(1, feats.dtype)
    while t < int(lengths[0]) and len(steps) < 400:
        tok, dec, h, c = model._tdt_step(feats[:, t:t + 1], mx.array([[last]], dtype=mx.int32), hidden[0], hidden[1])
        mx.eval(tok, dec, h, c)
        tok, dur = int(tok), model.durations[int(dec)]
        steps.append((t, tok, dur))
        if tok != model.blank_id:
            last, hidden = tok, (h, c)
        t += dur
        new += 1
        if dur != 0:
            new = 0
        elif model.max_symbols is not None and model.max_symbols <= new:
            t += 1
            new = 0
    print(label, "n_steps", len(steps), "non-blank", sum(1 for s in steps if s[1] != model.blank_id))
    print(label, "first steps", steps[:40])
