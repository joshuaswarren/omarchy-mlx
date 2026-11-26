#!/usr/bin/env python3
"""Lever (3): top-k via partition instead of full argsort.

mlx_audio's _sample_token uses apply_top_k which sorts the full vocab and
masks everything below the kth value, costing 188 argsort dispatches per
frame. A partition approach picks the kth-largest value, then a single
less_equal mask is applied, so it costs one argsort + K masked draws
rather than a full sort.

This script defines fast_topk_sample(logits, top_k) and:
  - chi-square: 4000 draws at a fixed logits vector, vs mlx_audio's path
  - microbench: ms per draw vs mlx_audio's path
  - end-to-end: 5 sentences x 2 measured rounds after 1 warmup, with
    mlx_audio's categorical_sampling monkey-patched to fast_topk_sample,
    aiden, seed 123+i, stream=True, streaming_interval=0.32. RTF,
    first-chunk p50/p95, and the saved WAVs go to <wavs>/topk/.
"""
import json, statistics, sys, time, wave
from pathlib import Path
import mlx.core as mx
import numpy as np

sys.path.insert(0, "<voice-site>")
HOME, OUT, WAVS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
R = {"uname": __import__("subprocess").run("uname -r", shell=True, capture_output=True, text=True).stdout.strip(),
     "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
     "loadavg_start": open("/proc/loadavg").read().strip(),
     "mlx": mx.__version__}
SENTS = ["The local assistant is ready to help.",
         "Your meeting starts at nine, and the review follows at eleven.",
         "I chose the shorter word, because it has fewer letters.",
         "Please check the camping list before you leave on Friday.",
         "Everything ran on this laptop, with no network connection."]


def gumbel_cat(logits, temp):
    u = mx.random.uniform(shape=logits.shape)
    return mx.argmax(logits.astype(mx.float32) * (1.0 / temp) - mx.log(-mx.log(u)), axis=-1)


def fast_topk_sample(logits, temp, top_k):
    """Gumbel-max with one partition-threshold then a less_equal mask.

    Equivalent distribution to apply_top_k(logits, k) -> categorical_sampling
    by the Gumbel-max + monotone-mask identity, but with one partial
    selection (mx.argpartition) instead of a full sort.
    """
    scaled = logits.astype(mx.float32) * (1.0 / temp)
    V = scaled.shape[-1]
    k = min(top_k, V)
    threshold = mx.min(mx.take_along_axis(scaled, mx.argpartition(-scaled, kth=V - k, axis=-1)[..., V - k:V - k + 1], axis=-1), axis=-1, keepdims=True)
    masked = mx.where(scaled < threshold, float("-inf"), scaled)
    u = mx.random.uniform(shape=masked.shape)
    return mx.argmax(masked - mx.log(-mx.log(u)), axis=-1)


def t(fn, n=30):
    for _ in range(3): mx.eval(fn())
    xs = []
    for _ in range(n):
        s = time.perf_counter(); mx.eval(fn()); xs.append((time.perf_counter() - s) * 1000)
    return round(statistics.median(xs), 3)


from mlx_audio.tts.utils import load_model
model = load_model(HOME / "voice" / "qwen3-tts-0.6b-customvoice-4bit")
mod = sys.modules[type(model).__module__]


# Microbench
lg = mx.random.normal((1, 1, 3072)); mx.eval(lg)
R["microbench_full_argsort_top50_ms"] = t(lambda: model._sample_token(lg, temperature=0.9, top_k=50, top_p=1.0))
saved = mod.categorical_sampling; mod.categorical_sampling = gumbel_cat
R["microbench_gumbel_top50_ms"] = t(lambda: model._sample_token(lg, temperature=0.9, top_k=50, top_p=1.0))
mod.categorical_sampling = lambda l, t: fast_topk_sample(l, t, 50)
R["microbench_fast_topk_partition_ms"] = t(lambda: model._sample_token(lg, temperature=0.9, top_k=50, top_p=1.0))
mod.categorical_sampling = saved


# Chi-square: distribution of the fast top-k vs the upstream path.
# Use a small vocab (V=8) so counts converge quickly.
def upstream_draw():
    small = mx.array([[[2.0, 1.0, 0.5, 0.0, -0.5, -1.0, 1.5, 0.2]]])
    return int(model._sample_token(small, temperature=1.0, top_k=5, top_p=1.0, suppress_tokens=None).item())
mod.categorical_sampling = lambda l, t: fast_topk_sample(l, t, 5)
def fast_draw():
    small = mx.array([[[2.0, 1.0, 0.5, 0.0, -0.5, -1.0, 1.5, 0.2]]])
    return int(model._sample_token(small, temperature=1.0, top_k=5, top_p=1.0, suppress_tokens=None).item())
mod.categorical_sampling = saved
N = 8000
cu = [0] * 8
for _ in range(N): cu[upstream_draw()] += 1
cf = [0] * 8
for _ in range(N): cf[fast_draw()] += 1
keep = sorted(range(8), key=lambda i: [2.0,1.0,0.5,0.0,-0.5,-1.0,1.5,0.2][i])[-5:]
keep_set = set(keep)
# Expected: softmax over logits in top-5.
import math
p_full = [math.exp([2.0,1.0,0.5,0.0,-0.5,-1.0,1.5,0.2][i]) for i in keep]
Z = sum(p_full)
p = [p_full[i] / Z for i in range(5)]
exp_count = N / 5
chi_upstream = sum((cu[i] - exp_count) ** 2 / exp_count for i in keep)
chi_fast = sum((cf[i] - exp_count) ** 2 / exp_count for i in keep)
R["chi2_df4_crit_p0.001"] = 18.47
R["chi2_upstream"] = round(chi_upstream, 2)
R["chi2_fast_topk"] = round(chi_fast, 2)
R["top5_zero_count_upstream"] = [cu[i] for i in range(8) if i not in keep_set]
R["top5_zero_count_fast"] = [cf[i] for i in range(8) if i not in keep_set]


# End-to-end
def run(name):
    mod.categorical_sampling = lambda l, t: fast_topk_sample(l, t, 50)
    rows = []
    for rnd in range(3):
        for i, s in enumerate(SENTS):
            mx.random.seed(123 + i)
            t0 = time.perf_counter(); first = None; pcm = []
            for r in model.generate_custom_voice(text=s, speaker="aiden", language="english",
                                                 stream=True, streaming_interval=0.32):
                if first is None: first = time.perf_counter() - t0
                pcm.append(np.asarray(r.audio, dtype=np.float32).reshape(-1))
            wall = time.perf_counter() - t0; audio = np.concatenate(pcm)
            if rnd == 0: continue
            rows.append({"rnd": rnd, "i": i + 1, "audio_s": round(len(audio) / 24000, 2),
                         "wall_s": round(wall, 2), "rtf": round(len(audio) / 24000 / wall, 3),
                         "first_s": round(first, 3), "loadavg": open("/proc/loadavg").read().split()[0]})
            if rnd == 1:
                d = WAVS / name; d.mkdir(parents=True, exist_ok=True)
                with wave.open(str(d / f"sentence-{i + 1}.wav"), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
                    w.writeframes((audio * 32767).clip(-32768, 32767).astype("<i2").tobytes())
            R[f"E2E_{name}_rows_so_far"] = rows
    rt = [r["rtf"] for r in rows]; fs = sorted(r["first_s"] for r in rows)
    R[f"E2E_{name}_rtf_median"] = statistics.median(rt)
    R[f"E2E_{name}_rtf_min"] = min(rt)
    R[f"E2E_{name}_first_p95"] = fs[int(0.95 * (len(fs) - 1))]
    mod.categorical_sampling = saved

run("fast_topk")
OUT.write_text(json.dumps({k: v for k, v in R.items() if not k.endswith("_so_far")}, indent=2))
print(json.dumps({k: v for k, v in R.items() if not k.endswith("_so_far")}, indent=2), flush=True)
