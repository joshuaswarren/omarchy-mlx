"""On-host workspace audit: measured activation high-water vs the analytic bound.

Runs the model at the frozen batch (default 16 questions, near-max context) in
the run dtype and compares the allocator's peak growth across the forward with
workspace_bytes()' analytic bound. On the CPU dev box fp16 emulation is slow —
this audit is intended to run ON the GPU host during the qualification window
(seconds), where it validates the admitted workspace term on real hardware.

    python tests/fixtures/laya/workspace_audit.py --model <converted ckpt> \
        [--max-questions 16] [--dtype float16] [--allow-cpu]

PASS rule: measured peak growth <= analytic workspace bound (the bound is
conservative; it over-counts the sliding window and counts fp32 logits).
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "serve"))

from mlx_omarchy_laya import server as laya_server  # noqa: E402
from mlx_omarchy_laya.api import LayaEngine  # noqa: E402
from mlx_omarchy_laya.model import load_encoder_config  # noqa: E402


def _memory_snapshots(mx):
    snap = {}
    for g in ("get_active_memory", "get_peak_memory"):
        if hasattr(mx, g):
            snap[g] = getattr(mx, g)()
    return snap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-questions", type=int, default=16)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    import mlx.core as mx

    manifest = json.loads((Path(args.model) / "manifest.json").read_text())
    enc_cfg = load_encoder_config(Path(args.model) / "encoder")
    bound = laya_server.workspace_bytes(manifest, enc_cfg, args.dtype, args.max_questions)
    params = laya_server.parameter_bytes(manifest, args.dtype)

    engine = LayaEngine(args.model, dtype=args.dtype, require_gpu=not args.allow_cpu)
    state = " ".join(
        "audit filler token %d padded state line for the workspace high-water measurement" % i
        for i in range(120)
    )
    questions = {}
    for i in range(args.max_questions):
        # exactly max_questions questions are served, so the bound's batch
        # argument matches the measured run (Main receipt review)
        if i % 2 == 0:
            questions["choice%d" % i] = {
                "type": "choice",
                "instructions": "Audit question %d: pick the matching queue with a moderately long instruction line." % i,
                "criteria": {"alpha": "first option", "beta": "second option",
                             "gamma": "third option", "delta": "fourth option", "omega": "everything else"},
            }
        else:
            questions["score%d" % i] = {
                "type": "score",
                "instructions": "Rate severity for audit item %d on the ordinal scale." % i,
                "criteria": ["none", "minor", "moderate", "major", "critical"],
            }

    before = _memory_snapshots(mx)
    t0 = time.perf_counter()
    out = engine.system_one(state, questions)
    wall = time.perf_counter() - t0
    after = _memory_snapshots(mx)

    growth = {k: after.get(k, 0) - before.get(k, 0) for k in before}
    measured_peak = growth.get("get_peak_memory", max(growth.values(), default=0))
    ok = measured_peak <= bound
    print(json.dumps({
        "dtype": args.dtype,
        "questions": len(questions),
        "input_tokens": out["usage"]["input_tokens"],
        "parameter_bytes": params,
        "analytic_workspace_bound": bound,
        "measured_peak_growth": measured_peak,
        "measured_growth_detail": growth,
        "forward_wall_s": round(wall, 3),
        "bound_conservative": ok,
        "verdict": "PASS" if ok else "MEASURED EXCEEDS BOUND",
    }, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
