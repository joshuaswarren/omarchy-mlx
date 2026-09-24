# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Offline validation of the device-chained control walk (no GPU).

Transcribes the slot schedule + GLSL control walk of
``vulkan_tdt_chain`` to Python and compares it against
``parakeet_tdt.greedy_tdt_decode`` (the pinned reference) over 300
randomized worlds that exercise the hard paths:

  * blank-hop runs whose cumulative advance exits the 7-row window
    (joint-only fallback slots),
  * duration-0 emission chains into the max-symbol roll-over
    (frame += 1 without advancing),
  * emissions in the final window rows (done mid-window),
  * duration-4 hops landing back inside the window.

The world is a PURE function of (state_epoch, frame) — both
implementations query identical cells, so any divergence is a
control-flow bug.  Run: python sim_chain_walk.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "overlay" / "tools"))

from coreml.parakeet_tdt import DecoderStep, JointDecision, greedy_tdt_decode
from coreml.reference import TdtConfig

DURATIONS = [0, 1, 2, 3, 4]


def decide(seed: int, vocab: int, blank: int, epoch: int, frame: int):
    """Pure (epoch, frame) -> (token_id, duration_index)."""
    h = (epoch * 2654435761 + frame * 40503 + seed * 9176) & 0xFFFFFFFF
    r = h % 100
    # forced long blank stretches: cumulative hops exit the window
    if frame % 23 == 0 and r < 80:
        return blank, 4
    # dur-0 emission chains: up to maxsym in a row at marked frames
    if frame % 37 == 0 and (epoch % 13) < 10 and r < 90:
        token = blank + 1 + (h >> 8) % (vocab - 1)
        return token, 0
    if r < 30:
        token = blank + 1 + (h >> 8) % (vocab - 1)
        duri = (h >> 20) % len(DURATIONS)
        return token, duri
    duri = (h >> 12) % len(DURATIONS)
    return blank, duri


def reference_decode(seed: int, vocab: int, blank: int, valid_frames: int,
                     config: TdtConfig):
    state = {"epoch": 0}

    def run_decoder(token, hidden, cell):
        state["epoch"] += 1
        return DecoderStep(f"dec{state['epoch']}", f"h{state['epoch']}",
                           f"c{state['epoch']}")

    def run_joint(frame, decoder_state):
        tok, duri = decide(seed, vocab, blank, state["epoch"], frame)
        return JointDecision(tok, duri)

    return greedy_tdt_decode(
        valid_frames=valid_frames, config=config,
        initial_hidden="h0", initial_cell="c0",
        run_decoder=run_decoder, run_joint=run_joint,
    )


def chain_decode(seed: int, vocab: int, blank: int, valid_frames: int,
                 config: TdtConfig, window_rows: int = 7):
    """Python transcription of the slot schedule + GLSL control walk."""
    maxsym = config.max_symbols_per_step
    frame, sym, count = 0, 0, 0
    epoch = 0
    state_valid = False
    token = blank
    emissions = []
    while frame < valid_frames:
        # slot: run the decoder unless the previous window was exhausted
        if not state_valid or token != blank:
            epoch += 1
            state_valid = True
        base = frame
        rows = []
        for r in range(window_rows):
            if base + r < valid_frames:
                tok, duri = decide(seed, vocab, blank, epoch, base + r)
                rows.append((tok, duri, DURATIONS[duri]))
            else:
                rows.append(None)
        # --- control walk (transcription of _CONTROL_BODY)
        r = 0
        need = False
        adv = False
        done = False
        err = 0
        while sym < maxsym:
            if frame >= valid_frames:
                done = True
                break
            if r >= window_rows:
                need = True
                break
            tok, duri, dval = rows[r]
            if duri < 0 or duri >= len(config.durations):
                err = 1
                break
            if tok == blank:
                hop = dval if dval > 1 else 1
                frame += hop
                if count != 0:
                    # state changes at every frame entry once a token has
                    # been emitted: next slot must re-step, rows are stale
                    sym = 0
                    break
                r += hop
                adv = True
                sym = 0
                continue
            if count >= valid_frames * maxsym:
                err = 2
                break
            emissions.append((tok, frame, dval))
            count += 1
            token = tok
            sym += 1
            if dval > 0:
                frame += dval
                adv = True
                sym = 0
                break
            break  # dur 0: next slot re-runs the decoder at this frame
        if err == 0 and not done and not adv and sym >= maxsym:
            frame += 1
            sym = 0
        if done or err:
            break
        if need:
            continue  # joint-only slot: state unchanged, fresh window
    return emissions


def main() -> int:
    vocab, blank = 64, 0
    failures = 0
    exhaustion_seen = 0
    rollover_seen = 0
    for seed in range(300):
        rng = random.Random(10_000 + seed)
        valid_frames = rng.randint(30, 400)
        config = TdtConfig(
            vocab_size=vocab, blank_token_id=blank,
            durations=DURATIONS, max_symbols_per_step=10,
        )
        ref = reference_decode(seed, vocab, blank, valid_frames, config)
        want = list(zip(ref.token_ids, ref.frame_indices, ref.durations))
        got = chain_decode(seed, vocab, blank, valid_frames, config)
        if got != want:
            failures += 1
            print(f"seed {seed}: MISMATCH frames={valid_frames} "
                  f"ref={len(want)} got={len(got)}")
            for i, (a, b) in enumerate(zip(want, got)):
                if a != b:
                    print(f"  first divergence at {i}: ref={a} got={b}")
                    break
            if failures > 3:
                return 1
        if any(d == 0 for _, _, d in want):
            rollover_seen += 1
        # window exhaustion exercised when a blank stretch spans > 7 frames
        blank_run = 0
        max_run = 0
        prev = None
        for _, fr, _ in want:
            _ = fr
        e, f = 0, 0
        while f < valid_frames:
            e += 1
            tok, duri = decide(seed, vocab, blank, e, f)
            if tok == blank:
                blank_run += max(DURATIONS[duri], 1)
                max_run = max(max_run, blank_run)
                f += max(DURATIONS[duri], 1)
            else:
                blank_run = 0
                f += DURATIONS[duri]
        if max_run > 7:
            exhaustion_seen += 1
    print(f"300 seeds: {'ALL MATCH' if failures == 0 else f'{failures} FAILURES'}"
          f" (walk coverage: emissions-with-dur0 present in {rollover_seen} seeds)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
