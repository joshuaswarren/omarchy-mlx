#!/usr/bin/env python3
"""TTS worker for the gdb CPU counter. Loads the Qwen3-TTS 0.6B CustomVoice
MLX model and generates 2 sentences. Signals completion via stdout so
the parent harness knows when to attach gdb.
"""
import os
import sys
import time


def announce(msg):
    print(msg, flush=True)


def main():
    announce("WORKER_LABEL=tts")
    home = os.environ.get("MLX_OMARCHY_HOME",
                          "<home>/agents/PairGates/homes/everyday")
    assets_dir = os.environ.get(
        "MLX_OMARCHY_TTS_ASSETS",
        "<home>/agents/PairGates/homes/everyday/voice/qwen3-tts-0.6b-customvoice-4bit")
    if not os.path.isdir(assets_dir):
        announce(f"WORKER_ASSETS_MISSING {assets_dir}")
        return

    from pathlib import Path
    from mlx_audio.tts.utils import load_model
    t0 = time.time()
    model = load_model(Path(assets_dir))
    t_load = time.time()
    announce(f"MODEL_LOAD_T={t_load-t0:.3f}")

    # Generate two sentences (slow; runs through the GPU dispatch).
    for i, text in enumerate(["Hello, this is a test sentence.",
                              "The quick brown fox jumps over the lazy dog."]):
        announce(f"WORKER_BEGIN_GEN i={i} text={text!r}")
        t_g0 = time.time()
        for result in model.generate_custom_voice(
                text=text, speaker="aiden", language="english",
                stream=False):
            pass
        t_g1 = time.time()
        announce(f"WORKER_END_GEN i={i} gen_t={t_g1-t_g0:.3f}")

    announce("WORKER_DONE")

    # Stay alive indefinitely so the parent can gdb-attach and let the
    # model sit in memory. Periodically re-generate so any CPU encoder
    # dispatches in the steady-state path would fire.
    while True:
        time.sleep(60)
        for text in ["stay-alive tick 1.", "stay-alive tick 2."]:
            for result in model.generate_custom_voice(
                    text=text, speaker="aiden", language="english", stream=False):
                pass


if __name__ == "__main__":
    main()