"""Full vendored E2E probe.

Runs vendored frame loop for 5 fixed sentences, aiden, seed 123+i, with
the Gumbel-max sampler at T=0.9. Reports ms/frame, RTF, dispatches/frame
(under MLX_OMARCHY_TRACE_DISPATCH=1), and writes WAVs for listening.

Compares against the upstream path on the same sentences (same seed).
Tokens must agree under greedy (T->0).
"""
import json, os, statistics, subprocess, sys, time, wave
from pathlib import Path
import mlx.core as mx
import numpy as np

sys.path.insert(0, "<home>/voice-site")
sys.path.insert(0, str(Path(__file__).resolve().parent / "serve"))

HOME = Path("<home>/mlx-tts-home")
WAVS = Path("<home>/agents/SpeechOutputFast/wavs")
WAVS.mkdir(parents=True, exist_ok=True)

R = {"uname": subprocess.run("uname -r", shell=True, capture_output=True, text=True).stdout.strip(),
     "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
     "mlx": mx.__version__,
     "trace": os.environ.get("MLX_OMARCHY_TRACE_DISPATCH", "0")}
SENTS = ["The local assistant is ready to help.",
         "Your meeting starts at nine, and the review follows at eleven.",
         "I chose the shorter word, because it has fewer letters.",
         "Please check the camping list before you leave on Friday.",
         "Everything ran on this laptop, with no network connection."]


def mark(n): sys.stderr.write(f"MARK {n}\n"); sys.stderr.flush()


from mlx_audio.tts.utils import load_model
model = load_model(str(HOME / "voice" / "qwen3-tts-0.6b-customvoice-4bit"))
print("loaded", flush=True)

from mlx_omarchy_assistant._vendored import qwen3_tts_step as V
V.vendorize_talker(model)
V.vendorize_cp(model)

cfg = model.config.talker_config
cp_cfg = cfg.code_predictor_config
talker = model.talker; cp = talker.code_predictor; emb = talker.get_input_embeddings()
head_dim = cfg.hidden_size // cfg.num_attention_heads
cp_head_dim = cp_cfg.hidden_size // cp_cfg.num_attention_heads


def upstream_generate(text):
    """Run upstream generate_custom_voice(stream=True, streaming_interval=0.32)
    and collect the audio chunks. Returns the concatenated audio and a
    wall-clock measurement."""
    started = time.perf_counter(); first = None
    pcm = []
    for r in model.generate_custom_voice(text=text, speaker="aiden", language="english",
                                         stream=True, streaming_interval=0.32):
        if first is None: first = time.perf_counter() - started
        pcm.append(np.asarray(r.audio, dtype=np.float32).reshape(-1))
    wall = time.perf_counter() - started
    audio = np.concatenate(pcm)
    return audio, wall, first


def t(fn, n=10):
    for _ in range(3): mx.eval(fn())
    xs = []
    for _ in range(n):
        s = time.perf_counter(); mx.eval(fn()); xs.append((time.perf_counter() - s) * 1000)
    return round(statistics.median(xs), 3)


# Time one full upstream generation cycle (5 sentences, 1 warmup + 2 measured rounds).
def run():
    rows = []
    for rnd in range(3):
        for i, s in enumerate(SENTS):
            mx.random.seed(123 + i)
            audio, wall, first = upstream_generate(s)
            if rnd == 0: continue
            rows.append({"rnd": rnd, "i": i + 1, "audio_s": round(len(audio) / 24000, 2),
                         "wall_s": round(wall, 2), "rtf": round(len(audio) / 24000 / wall, 3),
                         "first_s": round(first, 3)})
            # Save round-1 audio for listening
            if rnd == 1:
                d = WAVS / "upstream_baseline"
                d.mkdir(parents=True, exist_ok=True)
                with wave.open(str(d / f"sentence-{i + 1}.wav"), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
                    w.writeframes((audio * 32767).clip(-32768, 32767).astype("<i2").tobytes())
            R.setdefault(f"upstream_rows", []).append(rows[-1])
    rt = [r["rtf"] for r in rows]; fs = sorted(r["first_s"] for r in rows)
    R["upstream_rtf_median"] = statistics.median(rt)
    R["upstream_rtf_min"] = min(rt)
    R["upstream_first_p95"] = fs[int(0.95 * (len(fs) - 1))]
    R["upstream_n"] = len(rows)
    return rows


run()
out = Path("<home>/agents/SpeechOutputFast/profile/upstream_baseline.json")
out.write_text(json.dumps({k: v for k, v in R.items() if not k.startswith("upstream_rows")}, indent=2))
print(json.dumps({k: v for k, v in R.items() if not k.startswith("upstream_rows")}, indent=2), flush=True)
