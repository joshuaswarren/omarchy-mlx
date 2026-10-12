#!/usr/bin/env python3
"""KV state-slice eval-copy location probe (replaces kvcopy_decompose.py).

Answers, on the current tree, where the per-read KV state-slice copy is
produced and what the taken-read vs live-view contract is, with
self-verifying instrumentation: every stage must capture profile events,
and every stage carries an explicit verdict so silent zeros cannot pass
(the failure mode that retired kvcopy_decompose.py, which reported 0
copies in every stage without noticing).

Stages (synthetic arrays, cache [1,2,256,64] f16, offset 41):
  s1  slice-update write only        -> paste copies only (n=128); NO
                                        state-extent copy
  s2  state slice evaluated alone    -> CopyGeneralF16 n = offset*128
                                        fires with NO consumer: the copy
                                        lives at slice-eval time (the
                                        omarchy eval() densifier), not in
                                        any consumer. After the eval
                                        densifier is removed this stage
                                        records no commands at all; pass
                                        --expect-no-eval-copy to make
                                        absence (including an empty
                                        window) the expected verdict.
  s3  state read -> eager matmul     -> state-copy count equals the s2
                                        slice-eval baseline (the stage
                                        re-creates its slices); the
                                        consumer adds no copy of its own
  s4  state read -> fast SDPA        -> same as s3

Contract checks (bounded, synthetic), with the expectation built from
the KNOWN writes in numpy - the live view is never read, densified, or
otherwise evaluated until after the later writes, so the check cannot
pass on an early snapshot:
  live view    a lazy state view covering the next-write slot is held
               across subsequent slice-update writes and evaluated only
               afterwards; it must still show the pre-write values, and
               the writes must land in the parent (SliceUpdate allocates
               a fresh buffer).
  taken read   a state read taken and evaluated is a snapshot: later
               cache writes never rewrite it (verified over the FULL
               array against an independently built expectation, not a
               partial aggregate).

This probe is bounded to synthetic arrays; there is no model leg (model
benchmarks live in scripts/bench_decode.py). No backend behavior is
changed by this script.

Usage:
  MLX_OMARCHY_ALLOW_NON_APPLE=1 MLX_DISABLE_COMPILE=1 \
    python3 scripts/kv_state_views.py [--expect-no-eval-copy]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

OFF = 41

CHILD = r'''
import json, os, time
import mlx.core as mx

prof_path = os.environ["KV_STATE_VIEWS_PROF"]
mark_path = os.environ["KV_STATE_VIEWS_MARK"]

mf = open(mark_path, "w")
def mark(p):
    mf.write(json.dumps({"t": time.monotonic_ns(), "p": p}) + "\n")
    mf.flush()

off = 41
cache_k = mx.zeros((1, 2, 256, 64), mx.float16)
cache_v = mx.zeros((1, 2, 256, 64), mx.float16)
new_k = mx.ones((1, 2, 1, 64), mx.float16)
new_v = mx.ones((1, 2, 1, 64), mx.float16)

mark("s1_start")
cache_k[..., off-1:off, :] = new_k
cache_v[..., off-1:off, :] = new_v
mx.eval(cache_k, cache_v)
mark("s1_done")

mark("s2_start")
state_k = cache_k[..., :off, :]
state_v = cache_v[..., :off, :]
mx.eval(state_k, state_v)
mark("s2_done")

mark("s3_start")
state_k = cache_k[..., :off, :]
state_v = cache_v[..., :off, :]
q = mx.ones((1, 2, 1, 64), mx.float16)
scores = (q @ state_k.transpose(0, 1, 3, 2))
probs = mx.softmax(scores, axis=-1)
out = probs @ state_v
mx.eval(out)
mark("s3_done")

mark("s4_start")
state_k = cache_k[..., :off, :]
state_v = cache_v[..., :off, :]
q14 = mx.ones((1, 14, 1, 64), mx.float16) * 0.5
o = mx.fast.scaled_dot_product_attention(q14, state_k, state_v, scale=0.125)
mx.eval(o)
mark("s4_done")
print("stages done")
'''

CONTRACT = r'''
import mlx.core as mx, numpy as np
kv, S, d = 2, 64, 64
c = mx.zeros((1, kv, S, d), mx.float16)
for t in range(24):
    c[..., t:t+1, :] = mx.ones((1, kv, 1, d), mx.float16) * (t + 1)

# The live view: created LAZY and never evaluated, densified, or read
# until AFTER the later writes below.
view = c[..., :25, :]                       # covers the next-write slot

# Later writes. Slot 24 is INSIDE the held view's range: if slice-update
# ever mutated the parent buffer in place, the view would see it.
c[..., 24:25, :] = mx.ones((1, kv, 1, d), mx.float16) * 99.0
c[..., 25:26, :] = mx.ones((1, kv, 1, d), mx.float16) * 100.0

mx.eval(view)                               # first and only touch of view
post = np.asarray(view, dtype=np.float32)   # values as the view sees them

# Independent expectation built from the KNOWN writes alone: slots
# 0..23 hold t+1, slot 24 was never written before the view was taken
# so it must still read 0 inside the view.
expected = np.zeros((1, kv, 25, d), dtype=np.float32)
for t in range(24):
    expected[0, :, t, :] = t + 1.0
live_ok = bool(np.array_equal(post, expected))

# Taken read snapshot: what the read captured must not change later.
# Expectation built from the KNOWN writes alone (slots 0..19 hold t+1),
# compared over the FULL array - a partial max() would pass even if
# single overwritten elements regressed.
frozen = c[..., :20, :]
mx.eval(frozen)
c[..., 19:20, :] = mx.ones((1, kv, 1, d), mx.float16) * -1.0
mx.eval(c)
frozen_after = np.asarray(frozen, dtype=np.float32)
expected_frozen = np.zeros((1, kv, 20, d), dtype=np.float32)
for t in range(20):
    expected_frozen[0, :, t, :] = t + 1.0
frozen_ok = bool(np.array_equal(frozen_after, expected_frozen))
parent_ok = bool((np.asarray(c, dtype=np.float32)[0, :, 19, :] == -1.0).all())
slot24_ok = bool((np.asarray(c, dtype=np.float32)[0, :, 24, :] == 99.0).all())
print("RESULT", "LIVE", str(live_ok).lower(),
      "FROZEN", str(frozen_ok).lower(),
      "PARENT", str(parent_ok).lower(),
      "SLOT24", str(slot24_ok).lower())
'''


def kernel_names(compute_h):
    names, in_enum = [], False
    with open(compute_h) as fh:
        for line in fh:
            if "enum class ComputeKernel" in line:
                in_enum = True
                continue
            if in_enum:
                line = line.split("//", 1)[0]
                if "}" in line:
                    break
                name = line.strip().rstrip(",")
                if name and name != "Count" and " " not in name:
                    names.append(name)
    return names


def run_python(source, env_extra):
    env = dict(os.environ)
    env.update({
        "MLX_OMARCHY_ALLOW_NON_APPLE": "1",
        "MLX_DISABLE_COMPILE": "1",
    })
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", source], env=env,
                          capture_output=True, text=True)


def window_counters(prof, mark, names):
    sub_t, events = {}, []
    unresolved = 0
    for line in open(prof):
        ev = json.loads(line)
        if ev.get("k") == "s":
            sub_t[ev["s"]] = ev["t"]
        elif ev.get("k") == "d":
            if not isinstance(ev.get("e"), int) or ev["e"] < 0 or \
                    ev["e"] >= len(names):
                unresolved += 1
                continue
            events.append((sub_t.get(ev["s"], 0), ev))
    events.sort(key=lambda x: x[0])
    marks = [json.loads(l) for l in open(mark)]

    def stage_of(t):
        stage = None
        for m in marks:
            if m["t"] <= t:
                stage = m["p"]
        return stage

    per = defaultdict(Counter)
    for t, ev in events:
        per[stage_of(t)][f"{names[ev['e']]}/n={ev['n']}"] += 1
    return per, len(events), unresolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-no-eval-copy", action="store_true",
                    help="expectation for the tree AFTER the eval densifier "
                         "is removed: s2 records no commands at all "
                         "(empty window) and no state-extent copy")
    ap.add_argument("--compute-h",
                    default=".work/mlx/mlx/backend/omarchy/compute.h")
    args = ap.parse_args()

    try:
        names = kernel_names(args.compute_h)
    except OSError as e:
        print("FATAL: cannot read kernel table:", e)
        return 1
    if not names:
        print("FATAL: no kernel names parsed from", args.compute_h)
        return 1

    failures = []
    extent = OFF * 128  # 2 heads * off * 64 elements

    prof = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
    mark = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
    r = run_python(CHILD, {
        "MLX_OMARCHY_GPU_PROFILE": prof,
        "KV_STATE_VIEWS_PROF": prof,
        "KV_STATE_VIEWS_MARK": mark,
    })
    if r.returncode != 0:
        print("FATAL: stage child failed:", r.stderr[-500:])
        return 1
    trace = [l for l in r.stderr.splitlines() if "[materialize]" in l]
    per, total_events, unresolved = window_counters(prof, mark, names)
    if unresolved:
        print(f"FATAL: {unresolved} dispatch events reference kernel "
              f"indices outside the parsed compute.h table - the kernel "
              f"name mapping is unreliable, refusing to attribute copies")
        return 1
    if total_events == 0:
        print("FATAL: no profile events captured - instrumentation broken,"
              " refusing to report zeros")
        return 1
    print(f"stages instrumented: {total_events} dispatch events")

    def state_copy_count(window):
        return sum(n for key, n in per.get(window, {}).items()
                   if key.startswith("CopyGeneral") and f"n={extent}" in key)

    def window_total(window):
        return sum(per.get(window, {}).values())

    # s1: write only -> pastes, no state-extent copy. Instrumentation
    # must always be present here.
    if window_total("s1_start") == 0:
        failures.append("s1: empty window (instrumentation failure)")
    elif not any(k.startswith("CopyGeneral")
                 for k in per.get("s1_start", {})):
        failures.append("s1: no recognized CopyGeneral paste in the write "
                        "window - kernel name table shifted? (attribution "
                        "canary)")
    elif state_copy_count("s1_start"):
        failures.append("s1: unexpected state-extent copy on write-only")
    else:
        print(f"[PASS] s1 write-only: no state-extent copy "
              f"({dict(per.get('s1_start', {}))})")

    # s2: state slice evaluated ALONE. With the densifier present the
    # window holds the eval-copy; after densifier removal the slices
    # record no commands and the window is legitimately EMPTY - that is
    # the expected verdict only under --expect-no-eval-copy.
    s2 = state_copy_count("s2_start")
    if args.expect_no_eval_copy:
        if state_copy_count("s2_start") == 0:
            print("[PASS] s2 eval-alone: no state-extent copy recorded "
                  "(densifier absent, as expected)")
        else:
            failures.append(f"s2: expected no eval-copy, saw {s2} state "
                            "copies - densifier still present?")
    elif window_total("s2_start") == 0:
        failures.append("s2: empty window - densifier removed? rerun with "
                        "--expect-no-eval-copy")
    elif s2 >= 2:
        print(f"[PASS] s2 eval-alone: {s2}x CopyGeneral n={extent} with NO "
              f"consumer -> copy location = slice eval (omarchy eval() "
              f"densifier)")
    elif s2 == 0:
        failures.append("s2: no state-extent copy at slice-eval - densifier "
                        "removed or changed? rerun with --expect-no-eval-copy")
    else:
        failures.append(f"s2: unexpected single state copy ({s2})")

    # s3/s4: the stages re-create their slices, so their windows contain
    # their own slice-eval copies; the verdict is that the consumer adds
    # NOTHING beyond that per-slice eval cost. These windows must always
    # be non-empty: a consumer runs regardless of the densifier.
    baseline = s2
    for tag in ("s3_start", "s4_start"):
        if window_total(tag) == 0:
            failures.append(f"{tag}: empty window (instrumentation failure)")
            continue
        extra = state_copy_count(tag)
        if extra == baseline:
            print(f"[PASS] {tag}: state-copy count {extra} == slice-eval "
                  f"baseline {baseline}; consumer adds no copy of its own")
        else:
            failures.append(f"{tag}: state-copy count {extra} != slice-eval "
                            f"baseline {baseline} - consumer-side copy "
                            "appeared (eval-copy moved into the consumer)")

    consumer_mat = len(trace)
    print(f"consumer-side operand materializations: {consumer_mat} "
          "(expected 0: state slices ride strides into stride-tolerant consumers)")
    if consumer_mat:
        failures.append(f"consumer materialized state operands "
                        f"({consumer_mat} lines) - copy location moved")

    print("== contract: live view vs taken read ==")
    rc = run_python(CONTRACT, {})
    line = [l for l in rc.stdout.splitlines() if l.startswith("RESULT")]
    if rc.returncode != 0 or not line:
        failures.append(f"contract child failed: {rc.stderr[-300:]}")
    else:
        parts = line[0].split()
        live_ok, frozen_ok = parts[2], parts[4]
        parent_ok, slot24_ok = parts[6], parts[8]
        if live_ok == "true":
            print("[PASS] live view: lazy view held across later writes "
                  "reads its pre-write values at first evaluation "
                  "(fresh-buffer slice-update)")
        else:
            failures.append("live view corrupted by later writes - in-place "
                            "mutation detected")
        if frozen_ok == "true":
            print("[PASS] taken read: the state read is a snapshot; later "
                  "cache writes do not rewrite it (full-array check)")
        else:
            failures.append("state read not frozen - aliasing regression")
        if parent_ok == "true":
            print("[PASS] parent: later write landed in the parent cache")
        else:
            failures.append("later write did not land in parent")
        if slot24_ok == "true":
            print("[PASS] parent: in-range write visible to fresh reads "
                  "of the parent")
        else:
            failures.append("in-range write not visible in parent")

    if failures:
        print(f"{len(failures)} FAILURES")
        for f in failures:
            print(" -", f)
        return 1
    print("ALL VERDICTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
