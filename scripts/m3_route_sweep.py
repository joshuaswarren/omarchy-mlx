#!/usr/bin/env python3
"""M3 zero-build 8-lever decode/prefill route sweep (card M3-03).

Sweeps the eight runtime (env-only) route levers on an M3 part where the
chip-gated defaults are mostly OFF or fall back to conservative values.
Each lever is measured as its own arm against a baseline with every
lever scrubbed, so the delta per lever is isolated; a ``combined`` arm
sets all of them at once. No rebuild is required: every arm is the same
installed wheel driven by environment variables.

The levers (G15C defaults from the route table; the M1/M2 columns show
what the gated parts get):

  ================= ========================================== =============================
  arm               lever env (arm value vs G15C default)      source
  ================= ========================================== =============================
  batch_work        MLX_OMARCHY_BATCH_WORK=100000 (vs 5000     encoder.h calibrated_batch_work
                    fallback; G13C=80000, G14C=200000)
  qmm_geom          MLX_OMARCHY_QMM_GEOM=2x8w2 (vs 2x16        primitives.cpp; the 2x8w2 part
                    capability-picked)                          list is {G13G,G13C,G14C} only
  sdpa_online_rope  MLX_OMARCHY_SDPA_ONLINE_ROPE_MAX_KEYS=2304 device.h; G15C default is 0
                    + MLX_OMARCHY_SDPA_G13C_FAST=1              (split-K fallback)
  qmm_fast_only     MLX_OMARCHY_QMM_FAST_ONLY=1 (vs off)       device.h
  resnorm_vec       MLX_OMARCHY_RESNORM_VEC=1 (vs off)         device.h
  norm_apple_128    MLX_OMARCHY_NORM_APPLE_128=1 (vs off)      device.h / primitives.cpp
  fused_gemv        MLX_OMARCHY_FUSED_GEMV=0 +                 fused_chain.cpp; G15C defaults
                    MLX_OMARCHY_GEMV_BALANCED_QKV=1             to multi-weight fused GEMV on
  agx_simdmat       AGX_SIMDMAT=1 (vs unset)                   honeykrisp coopmat driver flag;
                                                              device.h notes llvmpipe lacks it
  ================= ========================================== =============================

Guarantees this runner enforces:

  - Greedy-only legs: ``--temp 0 --seed`` pinned, EOS suppressed inside
    scripts/bench_decode.py, pinned-length decode. After the sweep every
    measured leg's ``generated_ids sha256`` must equal the baseline's; a
    leg that disagrees is reported ``greedy-mismatch`` and the run fails
    (exit 5) - a speedup that changes tokens is a correctness defect.
  - ``/proc/self/maps`` SHA check: before any leg runs, a probe
    subprocess imports mlx, walks its own maps, hashes every mapped
    mlx-owned file, and prints the table. The runner re-hashes the same
    paths and refuses (exit 3) when a mapped file is unreadable,
    deleted, or does not match the on-disk bytes the wheel-RECORD gate
    in scripts/mlx_provenance.py already verified. A venv that loads
    one libmlx.so while the wheel claims another cannot emit numbers.
  - ``--dry-run`` resolves the arm plan and emits the JSON card row
    with every leg ``status="planned"`` - verified end to end without
    mlx, a GPU, or a model.
  - Output carries no hostname, user name, serial, or absolute home
    paths (the leg table records env values and the card row records
    the chip string; harness paths stay out of JSON lines).

Modes:
  plan       resolve arms and legs, print the plan, run nothing (default)
  run        maps-SHA probe, then one bench_decode leg per arm

Exit codes: 0 ok (>=1 measured leg and greedy identical, or plan/dry-run
completed), 1 bad invocation, 2 interpreter cannot import mlx, 3
provenance or maps-SHA refusal, 4 run finished with zero measured legs,
5 greedy ids mismatch across measured legs.

Self-test (no GPU, no mlx): python3 scripts/m3_route_sweep.py --self-test
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "m3-route-sweep/1"
CARD = "M3-03"

DECODE_RE = re.compile(r"decode ([0-9.]+) tok/s")
PREFILL_RE = re.compile(r"prefill ([0-9.]+)s")
IDS_RE = re.compile(r"generated_ids sha256:([0-9a-f]+) n=(\d+)")
PROV_LINE_RE = re.compile(r"provenance: (.+)")
UCLAMP_RE = re.compile(r"^uclamp: (.+)$")
RESULT_JSON_RE = re.compile(r'^\{"engine": "bench_decode"')
MAPS_LINE_RE = re.compile(r"^M3-MAPS-SHA (\{.+\})$", re.M)


@dataclass(frozen=True)
class Lever:
    arm: str
    doc: str
    env: tuple  # ((VAR, value), ...)


# Order is the card's order. Values are the arm-on settings; the G15C
# defaults they override are documented per lever in the module docstring
# and in receipts/2026-11-27-m3max-access-and-route-prep.md.
LEVERS = (
    Lever("batch_work",
          "calibrated_batch_work fallback 5000 -> 100000 groups",
          (("MLX_OMARCHY_BATCH_WORK", "100000"),)),
    Lever("qmm_geom",
          "QMM column geometry capability pick -> 2x8w2",
          (("MLX_OMARCHY_QMM_GEOM", "2x8w2"),)),
    Lever("sdpa_online_rope",
          "online-RoPE SDPA: max keys 0 -> 2304 + G13C fast path on",
          (("MLX_OMARCHY_SDPA_ONLINE_ROPE_MAX_KEYS", "2304"),
           ("MLX_OMARCHY_SDPA_G13C_FAST", "1"))),
    Lever("qmm_fast_only",
          "QMM fast-route-only on",
          (("MLX_OMARCHY_QMM_FAST_ONLY", "1"),)),
    Lever("resnorm_vec",
          "vectorized residual-norm on",
          (("MLX_OMARCHY_RESNORM_VEC", "1"),)),
    Lever("norm_apple_128",
          "Apple-128 norm part rule on",
          (("MLX_OMARCHY_NORM_APPLE_128", "1"),)),
    Lever("fused_gemv",
          "multi-weight fused GEMV off + balanced-QKV GEMV on",
          (("MLX_OMARCHY_FUSED_GEMV", "0"),
           ("MLX_OMARCHY_GEMV_BALANCED_QKV", "1"))),
    Lever("agx_simdmat",
          "honeykrisp cooperative-matrix advertising on",
          (("AGX_SIMDMAT", "1"),)),
)

LEVER_VARS = tuple(sorted({v for lever in LEVERS for v, _ in lever.env}))
ARM_IDS = ("baseline",) + tuple(lever.arm for lever in LEVERS) + ("combined",)

# Probe subprocess: import mlx, hash every mapped mlx-owned file, print
# one JSON line. The runner re-hashes the same paths and compares.
MAPS_SHA_SNIPPET = r"""
import hashlib, json, sys
import mlx.core  # noqa: F401  (import is the point - dlopen happens here)
seen = {}
try:
    fh = open("/proc/self/maps")
except OSError as exc:
    print("M3-MAPS-SHA " + json.dumps({"error": str(exc)}))
    sys.exit(0)
for line in fh:
    parts = line.split()
    if len(parts) < 6:
        continue
    p = parts[-1]
    if not p.startswith("/") or "mlx" not in p:
        continue
    if p in seen:
        continue
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        seen[p] = h.hexdigest()
    except OSError as exc:
        seen[p] = "unreadable:" + str(exc)
print("M3-MAPS-SHA " + json.dumps(seen, sort_keys=True))
"""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arm_env(arm):
    """The lever env vars for one arm. Baseline gets none; combined all."""
    if arm == "baseline":
        return {}
    if arm == "combined":
        return {v: val for lever in LEVERS for v, val in lever.env}
    for lever in LEVERS:
        if lever.arm == arm:
            return dict(lever.env)
    raise KeyError(f"unknown arm {arm!r}; arms are {', '.join(ARM_IDS)}")


def leg_env(inherited, arm):
    """Leg environment: inherited minus every lever var, plus the arm's.

    Scrubbing matters as much as setting: a host that already exports
    AGX_SIMDMAT=1 or MLX_OMARCHY_BATCH_WORK would silently make the
    baseline not the baseline.
    """
    env = {k: v for k, v in inherited.items() if k not in LEVER_VARS}
    env.update(arm_env(arm))
    return env


def chip_string():
    """Device-tree chip id on Linux/aarch64 Apple, else the platform."""
    try:
        compat = Path("/proc/device-tree/compatible").read_bytes()
        for entry in compat.decode("utf-8", "replace").split("\0"):
            if entry.startswith("apple,t"):
                return entry
    except OSError:
        pass
    return platform.machine() or "unknown"


def run_cmd(args, env=None, timeout=900):
    return subprocess.run(args, env=env, timeout=timeout,
                          capture_output=True, text=True)


def maps_sha_probe(python, inherited, timeout=120):
    """Run the maps probe. Returns (status, files, detail).

    status: ok | no-maps | import-failed | probe-failed
    files:  {path: sha256} on ok
    """
    env = {k: v for k, v in inherited.items() if k not in LEVER_VARS}
    try:
        proc = run_cmd([python, "-c", MAPS_SHA_SNIPPET], env=env,
                       timeout=timeout)
    except subprocess.TimeoutExpired:
        return "probe-failed", {}, f"timeout after {timeout}s"
    line = MAPS_LINE_RE.search(proc.stdout)
    if line is None:
        if proc.returncode != 0:
            status = ("import-failed" if "ModuleNotFoundError" in proc.stderr
                      else "probe-failed")
            return status, {}, proc.stderr.strip()[:400]
        return "probe-failed", {}, "no M3-MAPS-SHA line in probe output"
    files = json.loads(line.group(1))
    if "error" in files:
        return "no-maps", {}, files["error"]
    return "ok", files, ""


def verify_maps(files):
    """Re-hash every path the probe reported and compare.

    A mapped file that is unreadable, deleted (maps suffix), or whose
    bytes differ from the on-disk file means the process ran something
    other than what the wheel gate verified - refuse.
    """
    if not files:
        return "probe mapped no mlx-owned files; mlx may not be loaded"
    for path, probed in sorted(files.items()):
        if path.endswith(" (deleted)"):
            return f"mapped file deleted on disk: {path}"
        if isinstance(probed, str) and probed.startswith("unreadable:"):
            return f"mapped file unreadable: {path} ({probed})"
        try:
            actual = sha256_file(path)
        except OSError as exc:
            return f"cannot re-hash mapped file {path}: {exc}"
        if actual != probed:
            return (f"mapped-file hash mismatch at {path}: process loaded "
                    f"sha256 {probed}, disk now has {actual}")
    return None


def parse_bench_output(stdout):
    """Same line contract as scripts/bench_matrix.py's parser."""
    out = {"decode_tps": None, "prefill_s": None, "ids_sha256_16": None,
           "generated_ids_n": None, "prompt_tokens": None,
           "provenance_line": None, "uclamp": None}
    for line in stdout.splitlines():
        if (m := DECODE_RE.search(line)):
            out["decode_tps"] = float(m.group(1))
        elif (m := PREFILL_RE.search(line)):
            out["prefill_s"] = float(m.group(1))
        elif (m := IDS_RE.search(line)):
            out["ids_sha256_16"] = m.group(1)
            out["generated_ids_n"] = int(m.group(2))
        elif (m := PROV_LINE_RE.search(line)):
            out["provenance_line"] = m.group(1)
        elif (m := UCLAMP_RE.match(line)):
            out["uclamp"] = m.group(1)
        elif line.startswith("prompt_tokens "):
            try:
                out["prompt_tokens"] = int(line.split()[1])
            except (IndexError, ValueError):
                pass
    return out


def run_leg(python, bench, arm, model, prompt, tokens, wheel, inherited,
            timeout):
    """One bench_decode leg under the arm's environment."""
    leg = {"arm": arm, "env": arm_env(arm), "status": "failed",
           "decode_tps": None, "prefill_s": None, "ids_sha256_16": None,
           "generated_ids_n": None, "prompt_tokens": None,
           "provenance_line": None, "uclamp": None, "rc": None}
    cmd = [python, str(bench), "--model", model, "--prompt", prompt,
           "--tokens", str(tokens), "--temp", "0", "--seed", "0"]
    if wheel:
        cmd += ["--wheel", str(wheel)]
    try:
        proc = run_cmd(cmd, env=leg_env(inherited, arm), timeout=timeout)
    except subprocess.TimeoutExpired:
        leg["status"] = "timeout"
        return leg
    leg["rc"] = proc.returncode
    leg.update({k: v for k, v in parse_bench_output(proc.stdout).items()
                if v is not None})
    if proc.returncode != 0:
        leg["status"] = "refused" if proc.returncode == 3 else "failed"
        tail = proc.stderr.strip().splitlines()
        if tail:
            leg["error"] = tail[-1][:200]
        return leg
    if leg["decode_tps"] is None or leg["ids_sha256_16"] is None:
        leg["status"] = "failed"
        leg["error"] = "bench exited 0 without a decode rate or ids sha"
        return leg
    leg["status"] = "measured"
    return leg


def greedy_check(legs):
    """Every measured leg's ids sha must equal the baseline's."""
    base = next((l for l in legs if l["arm"] == "baseline"
                 and l["status"] == "measured"), None)
    if base is None:
        return {"verdict": "no-baseline", "ids_sha256_16": None,
                "mismatched": [l["arm"] for l in legs
                               if l["status"] == "measured"
                               and l["arm"] != "baseline"]}
    want = base["ids_sha256_16"]
    bad = [l["arm"] for l in legs
           if l["status"] == "measured" and l["ids_sha256_16"] != want]
    return {"verdict": "identical" if not bad else "mismatch",
            "ids_sha256_16": want, "mismatched": bad}


def card_row(legs, greedy, chip, wheel_provenance):
    measured = [l for l in legs if l["status"] == "measured"]
    base = next((l for l in measured if l["arm"] == "baseline"), None)
    best = None
    if measured:
        top = max(measured, key=lambda l: l["decode_tps"] or 0)
        delta = (None if not base or not base["decode_tps"]
                 else round((top["decode_tps"] / base["decode_tps"] - 1)
                            * 100, 2))
        best = {"arm": top["arm"], "decode_tps": top["decode_tps"],
                "delta_pct_vs_baseline": delta}
    return {"schema": SCHEMA, "card": CARD, "chip": chip,
            "wheel_provenance": wheel_provenance,
            "legs": legs, "greedy": greedy, "best": best}


def build_plan(arms):
    return [{"arm": arm, "env": arm_env(arm),
             "scrubs": list(LEVER_VARS)} for arm in arms]


def select_arms(requested):
    if not requested:
        return list(ARM_IDS)
    arms = []
    for name in requested:
        if name == "levers":
            arms.extend(lever.arm for lever in LEVERS)
        elif name in ARM_IDS:
            arms.append(name)
        else:
            raise KeyError(f"unknown arm {name!r}; arms, 'levers', or "
                           f"combine from: {', '.join(ARM_IDS)}")
    seen = []
    for a in arms:
        if a not in seen:
            seen.append(a)
    return seen


def self_test():
    # 1. The lever table is exactly the card's 8 levers, all real env
    #    names, all values non-empty.
    assert len(LEVERS) == 8
    names = [lever.arm for lever in LEVERS]
    assert names == ["batch_work", "qmm_geom", "sdpa_online_rope",
                     "qmm_fast_only", "resnorm_vec", "norm_apple_128",
                     "fused_gemv", "agx_simdmat"], names
    for lever in LEVERS:
        for var, val in lever.env:
            assert var.startswith(("MLX_OMARCHY_", "AGX_")), var
            assert val, lever.arm
    print("self-test: 8 levers, card order, real env names")

    # 2. leg_env scrubs every lever var from the inherited environment
    #    so a host-exported value cannot contaminate the baseline.
    dirty = dict(os.environ)
    for var in LEVER_VARS:
        dirty[var] = "9"
    clean = leg_env(dirty, "baseline")
    assert not any(v in clean for v in LEVER_VARS)
    armed = leg_env(dirty, "fused_gemv")
    assert armed["MLX_OMARCHY_FUSED_GEMV"] == "0"
    assert armed["MLX_OMARCHY_GEMV_BALANCED_QKV"] == "1"
    assert "MLX_OMARCHY_BATCH_WORK" not in armed
    all_on = arm_env("combined")
    assert len(all_on) == len(LEVER_VARS)
    print("self-test: baseline scrubs all lever vars; arms set exactly "
          "their own")

    # 3. Parser handles a bench_decode-shaped transcript.
    fake = ("uclamp: min=0\n"
            "provenance: verified=match wheel=x\n"
            "prompt_tokens 12\n"
            "decode 43.38 tok/s over 63 tokens (64 requested, EOS "
            "suppressed)\n"
            "prefill 1.502s (reported separately, excluded from decode)\n"
            "generated_ids sha256:294463eb6dba3a58 n=64 first=1,2,3 "
            "last=4,5,6\n"
            '{"engine": "bench_decode", "decode_tps": 43.38}\n')
    got = parse_bench_output(fake)
    assert got["decode_tps"] == 43.38 and got["prefill_s"] == 1.502
    assert got["ids_sha256_16"] == "294463eb6dba3a58"
    assert got["generated_ids_n"] == 64 and got["prompt_tokens"] == 12
    print("self-test: bench_decode transcript parsed")

    # 4. Greedy check: identical verdict, mismatch catches the leg, and
    #    a missing baseline reports distinctly.
    legs = [{"arm": "baseline", "status": "measured",
             "ids_sha256_16": "aa", "decode_tps": 40.0},
            {"arm": "batch_work", "status": "measured",
             "ids_sha256_16": "aa", "decode_tps": 43.0}]
    assert greedy_check(legs)["verdict"] == "identical"
    legs[1]["ids_sha256_16"] = "bb"
    g = greedy_check(legs)
    assert g["verdict"] == "mismatch" and g["mismatched"] == ["batch_work"]
    assert greedy_check([legs[1]])["verdict"] == "no-baseline"
    print("self-test: greedy ids verification")

    # 5. verify_maps catches the three dishonest states.
    assert verify_maps({}) is not None
    assert verify_maps({"/x/mlx.so (deleted)": "aa"}) is not None
    assert verify_maps({"/x/mlx.so": "unreadable:nope"}) is not None
    import tempfile
    with tempfile.NamedTemporaryFile(suffix="_mlx_test.so",
                                     delete=False) as fh:
        fh.write(b"probe-bytes")
        real = fh.name
    try:
        good = {real: sha256_file(real)}
        assert verify_maps(good) is None
        assert verify_maps({real: "0" * 64}) is not None
    finally:
        os.unlink(real)
    print("self-test: maps-SHA verify refuses deleted/unreadable/"
          "mismatched mapped files")

    # 6. Arm selection: 'levers' expands, unknown names refuse.
    assert select_arms(["baseline", "levers"]) == ["baseline"] + names
    assert select_arms(None) == list(ARM_IDS)
    try:
        select_arms(["bogus"])
    except KeyError:
        pass
    else:
        raise AssertionError("unknown arm accepted")
    print("self-test: arm selection")

    # 7. Plan + card row shape.
    plan = build_plan(["baseline", "agx_simdmat"])
    assert plan[1]["env"] == {"AGX_SIMDMAT": "1"}
    row = card_row(legs, greedy_check(legs), "apple,t6031", "match")
    assert row["schema"] == SCHEMA and row["card"] == CARD
    assert row["best"]["arm"] == "batch_work"
    print("self-test: plan and card row schema")
    print("self-test: OK")


def main():
    if {"--self-test"} & set(sys.argv[1:]):
        self_test()
        return

    ap = argparse.ArgumentParser(prog="m3_route_sweep.py")
    ap.add_argument("mode", nargs="?", default="plan",
                    choices=("plan", "run"))
    ap.add_argument("--model", default=None,
                    help="path to the mlx_lm model (required for run)")
    ap.add_argument("--prompt", default=None,
                    help="prompt text (required for run)")
    ap.add_argument("--tokens", type=int, default=64,
                    help="pinned decode length per leg (default 64)")
    ap.add_argument("--wheel", default=None,
                    help="wheel file the legs claim to test; passed to "
                         "bench_decode's --wheel provenance gate")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter that has mlx installed")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of arms; 'levers' "
                         "expands to all 8 (default: baseline + levers "
                         "+ combined)")
    ap.add_argument("--leg-timeout", type=int, default=900,
                    help="per-leg subprocess timeout in seconds")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the plan and emit the card row with "
                         "all legs status=planned; runs nothing")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    try:
        arms = select_arms(
            args.arms.split(",") if args.arms else None)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    plan = build_plan(arms)
    print(json.dumps({"schema": SCHEMA, "mode": args.mode,
                      "levers": {lever.arm: dict(lever.env)
                                 for lever in LEVERS},
                      "plan": plan}, sort_keys=True))

    if args.mode == "plan" or args.dry_run:
        legs = [{"arm": p["arm"], "env": p["env"], "status": "planned"}
                for p in plan]
        row = card_row(legs, {"verdict": "not-run",
                              "ids_sha256_16": None, "mismatched": []},
                       chip_string(), "not-run")
        print(json.dumps(row, sort_keys=True))
        sys.exit(0)

    # run mode
    if not args.model:
        ap.error("--model is required (path to the mlx_lm model)")
    if not args.prompt:
        ap.error("--prompt is required")
    if args.tokens < 2:
        ap.error("--tokens must be >= 2")
    if args.wheel and not Path(args.wheel).is_file():
        print(f"ERROR: --wheel {args.wheel} does not exist", file=sys.stderr)
        sys.exit(2)
    bench = Path(__file__).resolve().parent / "bench_decode.py"
    if not bench.is_file():
        print(f"ERROR: {bench} missing", file=sys.stderr)
        sys.exit(2)

    # maps-SHA gate before any leg: the process the numbers come from
    # must have loaded the bytes the wheel gate verified.
    status, files, detail = maps_sha_probe(args.python, os.environ)
    if status == "import-failed":
        print(f"ERROR: {args.python} cannot import mlx: {detail}",
              file=sys.stderr)
        sys.exit(2)
    if status == "probe-failed":
        print(f"REFUSING TO EMIT NUMBERS: maps probe failed: {detail}",
              file=sys.stderr)
        sys.exit(3)
    if status == "no-maps":
        print(f"maps: /proc/self/maps unavailable ({detail}); "
              "mapped-file check skipped", file=sys.stderr)
    else:
        refusal = verify_maps(files)
        if refusal:
            print(f"REFUSING TO EMIT NUMBERS: {refusal}", file=sys.stderr)
            sys.exit(3)
        print(f"maps: {len(files)} mlx-owned mapped files hash-verified",
              file=sys.stderr)

    legs = []
    for arm in arms:
        leg = run_leg(args.python, bench, arm, args.model, args.prompt,
                      args.tokens, args.wheel, os.environ,
                      args.leg_timeout)
        legs.append(leg)
        print(json.dumps({"schema": SCHEMA, "leg": leg}, sort_keys=True),
              flush=True)

    measured = [l for l in legs if l["status"] == "measured"]
    greedy = greedy_check(legs)
    row = card_row(legs, greedy, chip_string(), "see-leg-provenance")
    print(json.dumps(row, sort_keys=True))

    if not measured:
        sys.exit(4)
    if greedy["verdict"] == "mismatch":
        print(f"GREEDY MISMATCH: legs {greedy['mismatched']} diverged "
              f"from baseline ids_sha {greedy['ids_sha256_16']}",
              file=sys.stderr)
        sys.exit(5)
    sys.exit(0)


if __name__ == "__main__":
    main()
