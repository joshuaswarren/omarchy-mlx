# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Validate the device-chained TDT decode against the landed host path.

Runs on jwm1 with the installed venv's python and the worktree overlay on
sys.path.  Decodes the SAVED golden encoder tensor (no ANE, no mel) twice:

  1. host path  — ``greedy_tdt_decode`` with the fused ``run_step``
     callbacks (the landed b4757ac contract),
  2. chain path — ``vulkan_tdt_chain.run_tdt_chain``.

Compares token_ids, frame_indices, durations, hidden and cell bit-exact,
then times both (best of ``--reps``).  Usage:

  python validate_chain.py --cache-dir ~/.cache/mlx-omarchy/parakeet-reference/... \
      --encoder encoder_hidden.npy [--reps 5] [--chunks 64]

Exit 0 iff every check passes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--encoder", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--overlay", type=Path, required=True,
                    help="worktree overlay/tools directory for coreml imports")
    args = ap.parse_args()

    sys.path.insert(0, str(args.overlay))
    import mlx.core as mx  # noqa: E402

    from coreml.parakeet_tdt import (  # noqa: E402
        DecoderStep,
        JointDecision,
        greedy_tdt_decode,
    )
    from coreml.reference import ReferenceLock  # noqa: E402
    from coreml.vulkan_decoder import load_decoder  # noqa: E402
    from coreml.vulkan_decoder_step import (  # noqa: E402
        pack_step_weights,
        run_step,
    )
    from coreml.vulkan_tdt_chain import run_tdt_chain  # noqa: E402

    cache_dir = args.cache_dir
    lock = ReferenceLock.load(_lock_path(args.overlay))
    encoder_hidden = mx.array(np.load(args.encoder))
    valid_frames = int(encoder_hidden.shape[1])
    print(f"encoder {tuple(encoder_hidden.shape)} valid_frames={valid_frames}")

    decoder = load_decoder(cache_dir / "decoder.mlpackage")
    packed = pack_step_weights(decoder, cache_dir / "joint.mlpackage")
    config = lock.tdt

    with mx.stream(mx.gpu):
        hidden0 = mx.zeros((2, 1, 640), dtype=mx.float32)
        cell0 = mx.zeros((2, 1, 640), dtype=mx.float32)
    mx.eval(hidden0, cell0)

    frame_holder = [0]
    spec = {"base": None, "state": None, "host": None}
    fused = {"frame": None, "state": None, "tok": None, "dur": None}

    def run_decoder(token_id, current_hidden, current_cell):
        state_out, tok, dur, _, _, window = run_step(
            packed, current_hidden, current_cell, token_id,
            encoder_hidden, frame_holder[0],
            spec_frames=6, spec_valid=valid_frames,
        )
        mx.eval(state_out)
        dec_state = state_out[0:640].reshape(1, 640)
        fused["frame"] = frame_holder[0]
        fused["state"] = dec_state
        fused["tok"] = tok
        fused["dur"] = dur
        spec["base"] = frame_holder[0] + 1
        spec["state"] = dec_state
        spec["host"] = np.asarray(window) if window is not None else None
        return DecoderStep(
            dec_state,
            state_out[640:1920].reshape(2, 1, 640),
            state_out[1920:3200].reshape(2, 1, 640),
        )

    def run_joint(frame_index, decoder_state):
        frame_holder[0] = frame_index
        if fused["frame"] == frame_index and fused["state"] is decoder_state:
            return JointDecision(fused["tok"], fused["dur"])
        host = spec["host"]
        if host is not None and spec["state"] is decoder_state:
            offset = frame_index - spec["base"]
            if 0 <= offset < host.shape[0]:
                row = host[offset]
                return JointDecision(
                    int(np.argmax(row[:8193])),
                    int(np.argmax(row[8193:8198])),
                )
        state_out, tok, dur, _, _, window = run_step(
            packed, None, None, 0, encoder_hidden, frame_index,
            skip_lstm=True, dec_in=decoder_state,
            spec_frames=6, spec_valid=valid_frames,
        )
        mx.eval(state_out)
        spec["base"] = frame_index + 1
        spec["state"] = decoder_state
        spec["host"] = np.asarray(window) if window is not None else None
        return JointDecision(tok, dur)

    # ---- host reference (correctness anchor; also the incumbent timing)
    def host_pass():
        frame_holder[0] = 0
        spec.update(base=None, state=None, host=None)
        fused.update(frame=None, state=None, tok=None, dur=None)
        t0 = time.perf_counter()
        out = greedy_tdt_decode(
            valid_frames=valid_frames,
            config=config,
            initial_hidden=hidden0,
            initial_cell=cell0,
            run_decoder=run_decoder,
            run_joint=run_joint,
        )
        return out, (time.perf_counter() - t0) * 1e3

    ref, ref_ms = host_pass()
    print(f"host: {len(ref.token_ids)} emissions  {ref_ms:8.1f} ms  "
          f"sha={hashlib_lists(ref)}")

    # ---- chain candidate
    def chain_pass():
        with mx.stream(mx.gpu):
            hidden0c = mx.zeros((2, 1, 640), dtype=mx.float32)
            cell0c = mx.zeros((2, 1, 640), dtype=mx.float32)
        mx.eval(hidden0c, cell0c)
        t0 = time.perf_counter()
        got = run_tdt_chain(
            packed, encoder_hidden, valid_frames, config,
            hidden0c, cell0c, slots_per_chunk=args.chunk,
        )
        ms = (time.perf_counter() - t0) * 1e3
        _ = np.asarray(got.hidden)  # sync
        print(f"    chain slots_used={got.slots_used}")
        return got, ms

    got, chain_ms = chain_pass()

    checks = {
        "token_ids": got.token_ids == ref.token_ids,
        "frame_indices": got.frame_indices == ref.frame_indices,
        "durations": got.durations == ref.durations,
        "hidden": np.array_equal(np.asarray(got.hidden),
                                 np.asarray(ref.hidden)),
        "cell": np.array_equal(np.asarray(got.cell), np.asarray(ref.cell)),
    }
    for name, ok in checks.items():
        print(f"check {name}: {'PASS' if ok else 'FAIL'}")
    if not all(checks.values()):
        n = min(len(ref.token_ids), len(got.token_ids))
        div = next((i for i in range(n)
                    if (ref.token_ids[i], ref.frame_indices[i],
                        ref.durations[i])
                    != (got.token_ids[i], got.frame_indices[i],
                        got.durations[i])), n)
        print(f"MISMATCH: divergence at emission {div} of ref={len(ref.token_ids)} got={len(got.token_ids)}")
        lo = max(0, div - 3)
        for i in range(lo, min(n, div + 3)):
            print(f"  [{i}] ref=({ref.token_ids[i]},{ref.frame_indices[i]},{ref.durations[i]})"
                  f" got=({got.token_ids[i]},{got.frame_indices[i]},{got.durations[i]})")
        print("REF tail:", list(zip(ref.token_ids[-5:], ref.frame_indices[-5:], ref.durations[-5:])))
        print("GOT tail:", list(zip(got.token_ids[-5:], got.frame_indices[-5:], got.durations[-5:])))
        return 1

    best_chain = chain_ms
    best_host = ref_ms
    for _ in range(max(0, args.reps - 1)):
        _, ms = chain_pass()
        best_chain = min(best_chain, ms)
        _, ms = host_pass()
        best_host = min(best_host, ms)
    print(f"timing best-of-{args.reps}: host {best_host:8.1f} ms  "
          f"chain {best_chain:8.1f} ms  ({best_host / best_chain:.2f}x)")
    return 0


def _lock_path(overlay: Path) -> Path:
    return overlay / "coreml" / "parakeet-reference.lock"


def hashlib_lists(out) -> str:
    import hashlib

    blob = repr((out.token_ids, out.frame_indices, out.durations)).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
