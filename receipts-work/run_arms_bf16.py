#!/usr/bin/env python3
"""BF16 decode token attribution: ablation arms (fork driver).

Same method as receipts/2026-09-10-decode-attribution/run_arms.py, for
the BF16 model and the BF16 decode census classes. Baseline legs run the
pristine release wheel and assert the canonical BF16 fork digests; each
ablation arm runs the measurement wheel with MLX_OMARCHY_ABLATE set to
one class (plus "all") and asserts the digest FLIPPED (engagement gate)
and the wheel provenance matched. Classes: gemv, attn, cast, ewise,
copy, rope, rms, swiglu, sampler, all.

Run from the worktree root under the GPU lock on a quiet machine:
  timeout 3300 flock -w 3200 python3 \
    receipts/2026-09-11-bf16-decode-attribution/run_arms_bf16.py \
    --python .work/venv-run-bf16dec/bin/python \
    --ablate-python .work/venv-ablate-bf16dec/bin/python \
    --wheel <pristine release.whl> --ablate-wheel <ablate release.whl> \
    --arms baseline gemv attn cast ewise \
    --out receipts/2026-09-11-bf16-decode-attribution/arms
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
import sys

PIN = "56d07e766edd7159fbe12ed12d9cf114bf38bf1e"
EXPECTED = {
    "short-decode-32": "f26175202f3dabe9",
    "long-decode-128": "8690dc83246b39f8",
    "longctx-1024-decode-32": "ff502900d2a179a5",
}
ALL_LEGS = ["short-decode-32", "long-decode-128", "longctx-1024-decode-32"]
# arm -> reps per leg (default: 6 per leg for class arms)
ARMS = {
    "baseline": {w: 12 for w in ALL_LEGS},
    "gemv": {w: 6 for w in ALL_LEGS},
    "attn": {w: 6 for w in ALL_LEGS},
    "cast": {w: 6 for w in ALL_LEGS},
    "ewise": {w: 6 for w in ALL_LEGS},
    "copy": {w: 6 for w in ALL_LEGS},
    "rope": {w: 6 for w in ALL_LEGS},
    "rms": {w: 6 for w in ALL_LEGS},
    "swiglu": {w: 6 for w in ALL_LEGS},
    "sampler": {w: 6 for w in ALL_LEGS},
    "all": {w: 8 for w in ALL_LEGS},
    # pristine-wheel A/B of the dormant MLX_OMARCHY_SDPA_BF16_FAST env
    # (bf16-storage attention composition); measurement only.
    "bf16fast": {w: 6 for w in ALL_LEGS},
}


def loadavg():
    return float(open("/proc/loadavg").read().split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parents[2])
    ap.add_argument("--python", type=Path, required=True,
                    help="venv python holding the pristine release wheel")
    ap.add_argument("--ablate-python", type=Path, required=True,
                    help="venv python holding the ablation wheel")
    ap.add_argument("--wheel", type=Path, required=True)
    ap.add_argument("--ablate-wheel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="subset of arms to run (default all)")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    arms = {k: v for k, v in ARMS.items()
            if args.arms is None or k in args.arms}
    assert arms, "no arms selected"

    manifest = json.loads((root / "scripts/bench_matrix.json").read_text())
    manifest["models"] = [m for m in manifest["models"]
                          if m["id"] == "qwen25-0.5b-bf16"]
    manifest["generation"]["engine_script"] = "bench_decode_identity.py"
    manifest_path = out / "manifest-bf16.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    ndjson = (out / "legs.ndjson").open("a")
    max_rep = max(max(reps.values()) for reps in arms.values())
    for rep in range(1, max_rep + 1):
        for arm, reps in arms.items():
            for workload, nreps in reps.items():
                if rep > nreps:
                    continue
                t_leg = time.time()
                stem = f"r{rep}-{arm}-{workload}"
                out_json = out / f"{stem}.json"
                log_path = out / f"{stem}.log"
                if out_json.exists():
                    record_peak = None
                    print(f"{stem} cached", flush=True)
                else:
                    record_peak = 0.0
                    env = {k: v for k, v in os.environ.items()
                           if not k.startswith("MLX_")}
                    env.pop("VK_DRIVER_FILES", None)
                    env.pop("HK_PERFTEST", None)
                    env["HF_HUB_OFFLINE"] = "1"
                    env["MLX_DISABLE_COMPILE"] = "1"
                    pristine = arm in ("baseline", "bf16fast")
                    if arm == "bf16fast":
                        env["MLX_OMARCHY_SDPA_BF16_FAST"] = "1"
                    if not pristine:
                        env["MLX_OMARCHY_ABLATE"] = arm
                        env["MLX_OMARCHY_ABLATE_DEBUG"] = "1"
                    python = Path(os.path.abspath(
                        str(args.python if pristine else args.ablate_python)))
                    wheel = Path(os.path.abspath(
                        str(args.wheel if pristine else args.ablate_wheel)))
                    cmd = [
                        sys.executable,  # runner process: stdlib only
                        "scripts/bench_matrix.py", "--mode", "run",
                        "--manifest", str(manifest_path),
                        "--select", workload,
                        "--python", str(python),
                        "--wheel", str(wheel),
                        "--expect-pins", f"qwen25-0.5b-bf16={PIN}",
                        "--host-label", f"m1-test-host-bf16-decode-attrib-{arm}-r{rep}",
                        "--timeout", str(args.timeout),
                        "--out", str(out_json),
                    ]
                    peak = 0.0
                    record_peak = 0.0
                    with log_path.open("w") as log:
                        proc = subprocess.Popen(
                            cmd, cwd=root, env=env, stdout=log,
                            stderr=subprocess.STDOUT)
                        while proc.poll() is None:
                            peak = max(peak, loadavg())
                            time.sleep(5)
                        rc = proc.returncode
                    record_peak = round(peak, 2)
                    if rc != 0:
                        raise SystemExit(
                            f"{stem} FAILED rc={rc} peak_loadavg={record_peak}"
                            " (see log)")
                data = json.loads(out_json.read_text())
                assert data["clean_check"]["status"] == "clean", \
                    data["clean_check"]
                assert data["binary_provenance"]["omarchy"]["verified"] == \
                    "match", data["binary_provenance"]
                legs = [l for l in data["legs"]
                        if l["workload_id"] == workload]
                assert len(legs) == 1 and legs[0]["status"] == "measured", \
                    [(l["leg_id"], l["status"]) for l in data["legs"]]
                leg = legs[0]
                m = leg["metrics"]
                digest = m["generated_ids_sha256_16"]
                if arm == "baseline":
                    assert digest == EXPECTED[workload], (workload, digest)
                elif arm not in ("bf16fast", "sampler"):
                    # bf16fast keeps the pristine kernel set; its digest
                    # MAY equal canonical (gate may be bit-neutral on a
                    # leg) - record, never assert. sampler is
                    # digest-neutral by construction here: logsumexp and
                    # argreduce feed the reported logprobs, not the
                    # greedy token choice (mlx-lm temp=0 argmax does not
                    # consume either output), so ablation cannot flip
                    # the stream; the wall-time marginal is the point.
                    assert digest != EXPECTED[workload], \
                        (workload, digest, "ablation did not engage")
                record = {
                    "arm": arm, "rep": rep, "workload": workload,
                    "decode_tok_s": m["decode_tok_s"],
                    "prefill_s": m.get("prefill_s"),
                    "prompt_tokens": m.get("prompt_tokens"),
                    "digest": digest,
                    "peak_loadavg_1m": record_peak,
                    "duration_s": leg.get("duration_s"),
                    "host_label": leg.get("host_label"),
                    "wall_s": round(time.time() - t_leg, 1),
                }
                ndjson.write(json.dumps(record, sort_keys=True) + "\n")
                ndjson.flush()
                print(f"{stem} ok {m['decode_tok_s']} tok/s {digest} "
                      f"peak={record_peak}", flush=True)
    ndjson.close()
    print("ALL_ARMS_DONE", flush=True)


if __name__ == "__main__":
    main()
