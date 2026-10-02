#!/usr/bin/env python3
"""Gate 8 driver: Kokoro synth of the reference sentence through the
installed release wheel. The Synthesis worker is spawned (multiprocessing),
so ALL driver work must live under the __main__ guard. The voice-pack home
comes from MLX_OMARCHY_TTS_HOME (set by g8-kokoro.sh)."""
import io
import os
import sys
import threading
import time
import wave
from pathlib import Path


def main() -> int:
    from mlx_omarchy_assistant.synthesis import Synthesis

    sentence = "Your meeting starts at nine, and the review follows at eleven."
    home = Path(os.environ["MLX_OMARCHY_TTS_HOME"])
    s = Synthesis(home)
    print("PREPARE", s.prepare(True), flush=True)
    s.set_voice("af_heart")  # Kokoro engine

    def load_and_psi():
        load = open("/proc/loadavg").read().split()[:3]
        psi = {}
        for line in open("/proc/pressure/cpu"):
            if line.startswith("some"):
                for field in line.split()[1:]:
                    key, value = field.split("=")
                    psi[key] = value
        return "/".join(load), psi.get("avg10"), psi.get("avg60")

    before = load_and_psi()
    t0 = time.monotonic()
    wav = s.synthesize(sentence, threading.Event())
    wall = time.monotonic() - t0
    after = load_and_psi()

    with wave.open(io.BytesIO(wav)) as handle:
        rate = handle.getframerate()
        frames = handle.getnframes()
    seconds = frames / rate
    rtf = seconds / wall
    print(f"KOKORO_SMOKE audio_s={seconds:.3f} wall_s={wall:.3f} rtf={rtf:.3f}")
    print(f"LOAD before={before[0]} after={after[0]} "
          f"PSI_CPU_AVG10 before={before[1]} after={after[1]}")
    ok = 3.5 < seconds < 4.5 and wall < 5.5
    print("KOKORO_SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
