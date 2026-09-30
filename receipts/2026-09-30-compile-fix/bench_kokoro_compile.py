"""Kokoro sentence RTF: eager vs mx.compile of the Snake blocks (resumable).

usage: bench_kokoro_compile.py <repo> <kokoro-pack-dir> <out-dir>
Needs mlx_audio (Kokoro) and the G2P runtime importable (PYTHONPATH).

Variants run in one process, in order, on the same wheel (CF_VARIANTS=a,b
selects a subset; eager must run before the seeded compiled comparisons):
  eager       - no compile
  snake       - generator.resblocks + generator.noise_res (AdaINResBlock1,
                the Snake blocks) wrapped in mx.compile
  all_blocks  - snake + decoder.encode + decoder.decode[] (AdainResBlk1d)
Per variant: 1 warmup + 3 measured runs of one sentence, then one seeded
correctness run compared with the eager run of the same seed. Every item
is appended to results.jsonl with fsync; a restart skips finished items.
"""
import ctypes
import json
import os
import sys
import time
from pathlib import Path

REPO, PACK, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "serve"))
from mlx_provenance import installed_provenance, provenance_line  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

TEXT = "Your meeting starts at nine, and the review follows at eleven."
VOICE = "af_heart"
SEED = 1234
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "results.jsonl"
prov = provenance_line(installed_provenance())
print(prov, flush=True)
done = set()
if RESULTS.exists():
    for line in RESULTS.read_text().splitlines():
        rec = json.loads(line)
        done.add((rec["variant"], rec["item"]))


def record(rec):
    rec.update(provenance=prov, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
               loadavg=Path("/proc/loadavg").read_text().strip())
    with RESULTS.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    print(json.dumps(rec), flush=True)


class Snap(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "gpu_primitive_dispatches", "vk_submissions", "vk_buffer_copies",
        "vk_buffer_fills", "vk_compute_dispatches", "omarchy_finalize_calls",
        "commit_calls_with_work", "commit_calls_noop")]


with open("/proc/self/maps") as fh:
    LIB = ctypes.CDLL(next(line.split()[-1] for line in fh
                           if line.rstrip().endswith("/libmlx.so")))


def counters():
    s = Snap()
    LIB.mlx_omarchy_trace_snapshot(ctypes.byref(s))
    return s.gpu_primitive_dispatches, s.vk_compute_dispatches


import espeakng_loader  # noqa: E402
from phonemizer.backend.espeak.wrapper import EspeakWrapper  # noqa: E402

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
from mlx_omarchy_assistant.synthesis import _kokoro_install_trig_reduction  # noqa: E402

_kokoro_install_trig_reduction()
from mlx_audio.tts.utils import load_model  # noqa: E402
from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline  # noqa: E402
from mlx_audio.tts.models.kokoro.voice import load_voice_tensor  # noqa: E402

model = load_model(PACK)
pipe = KokoroPipeline(lang_code="a", model=model, repo_id=str(PACK))
pipe.voices[VOICE] = load_voice_tensor(str(PACK / "voices" / f"{VOICE}.safetensors"))
dec = model.decoder
gen = dec.generator
orig = {"resblocks": list(gen.resblocks), "noise_res": list(gen.noise_res),
        "encode": dec.encode, "decode": list(dec.decode)}


def configure(variant):
    snake = variant in ("snake", "all_blocks")
    blocks = variant == "all_blocks"
    wrap = mx.compile if snake else (lambda b: b)
    gen.resblocks = [wrap(b) for b in orig["resblocks"]]
    gen.noise_res = [wrap(b) for b in orig["noise_res"]]
    dec.encode = mx.compile(orig["encode"]) if blocks else orig["encode"]
    dec.decode = [mx.compile(b) for b in orig["decode"]] if blocks else list(orig["decode"])


def synth(seed=None):
    if seed is not None:
        mx.random.seed(seed)
    c0 = counters()
    t0 = time.perf_counter()
    parts = [np.asarray(r.audio, dtype=np.float32).reshape(-1)
             for r in pipe(TEXT, voice=VOICE, speed=1.0)]
    wall = time.perf_counter() - t0
    c1 = counters()
    audio = np.concatenate(parts)
    return audio, {"wall_s": wall, "audio_s": audio.shape[0] / 24000.0,
                   "rtf": audio.shape[0] / 24000.0 / wall,
                   "finite": bool(np.isfinite(audio).all()),
                   "gpu_primitive_dispatches": c1[0] - c0[0],
                   "vk_compute_dispatches": c1[1] - c0[1]}


REF = OUT / f"eager-seed{SEED}.npy"
for variant in os.environ.get("CF_VARIANTS", "eager,snake,all_blocks").split(","):
    items = ["warmup", "run1", "run2", "run3", "seeded"]
    if all((variant, i) in done for i in items):
        continue
    configure(variant)
    for item in items:
        if (variant, item) in done and item != "warmup":
            continue
        audio, rec = synth(SEED if item == "seeded" else None)
        rec.update(variant=variant, item=item)
        if item == "seeded":
            if variant == "eager":
                np.save(REF, audio)
                rec["max_abs_diff_vs_eager"] = 0.0
            else:
                ref = np.load(REF)
                n = min(ref.shape[0], audio.shape[0])
                diff = audio[:n] - ref[:n]
                rec["samples_match"] = bool(ref.shape[0] == audio.shape[0])
                rec["max_abs_diff_vs_eager"] = float(np.abs(diff).max())
                rec["snr_db_vs_eager"] = float(10 * np.log10(
                    np.sum(ref[:n] ** 2) / max(np.sum(diff ** 2), 1e-30)))
        if (variant, item) not in done:
            record(rec)
            done.add((variant, item))
        mx.clear_cache()
print("all items done", flush=True)
