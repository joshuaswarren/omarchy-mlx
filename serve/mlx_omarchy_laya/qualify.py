"""Hardware qualification runner for Laya on mlx-omarchy (Apple GPU).

Compares the served/GPU envelopes against the committed CPU reference
fixture (tests/fixtures/laya/laya_reference.json) and prints a
receipt-ready block: provenance line, device identity, per-case deltas,
latency, and a PASS/FAIL verdict against the documented gates.

Run on the qualification host (t6001-test-host/m1-test-host) AFTER Main opens the hardware
window, with the mlx-omarchy wheel (Vulkan GPU backend) active:

    python -m mlx_omarchy_laya.qualify --model ~/.local/share/mlx-omarchy/models/laya
    # or against a running server:
    python -m mlx_omarchy_laya.qualify --url http://127.0.0.1:8081 --model <ckpt>

Gates (fp16 GPU vs fp32 CPU reference, first-run calibration; the receipt
records the achieved deltas either way):
  - choice argmax agreement: 100% of questions
  - probability delta vs fixture: max <= 2e-2
  - act_probability delta:        max <= 5e-2
  - input token identity: exact
  - latency: recorded cold + warm, no gate on first qualification
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "laya" / "laya_reference.json"

ARGMAX_GATE = 1.0
PROB_GATE = 2e-2
ACT_GATE = 5e-2


def _provenance(repo_root: Path) -> str:
    script = repo_root / "scripts" / "mlx_provenance.py"
    if script.exists():
        try:
            out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
            return (out.stdout or out.stderr).strip().splitlines()[-1] if (out.stdout or out.stderr) else "provenance unavailable"
        except Exception as exc:
            return "provenance unavailable: %s" % exc
    return "provenance script not present"


def _run_case_direct(model, state, questions, require_gpu):
    from mlx_omarchy_laya.api import LayaEngine

    if not hasattr(_run_case_direct, "_engine") or _run_case_direct._model != model:
        _run_case_direct._engine = LayaEngine(model, dtype="float16", require_gpu=require_gpu)
        _run_case_direct._model = model
    t0 = time.perf_counter()
    out = _run_case_direct._engine.system_one(state, questions)
    return out, (time.perf_counter() - t0) * 1000.0


def _run_case_url(url, state, questions):
    req = urllib.request.Request(url.rstrip("/") + "/v1/decisions",
                                 data=json.dumps({"state": state, "questions": questions}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return out, (time.perf_counter() - t0) * 1000.0


def _memory_snapshot():
    """Best-effort resident/peak memory in bytes across mlx builds."""
    import importlib

    import mlx.core as mx

    snap = {}
    try:
        metal = importlib.import_module("mlx.metal")
        for g in ("get_active_memory", "get_peak_memory", "get_cache_memory"):
            if hasattr(metal, g):
                snap[g] = getattr(metal, g)()
    except Exception:
        pass
    for g in ("get_active_memory", "get_peak_memory"):
        if g not in snap and hasattr(mx, g):
            try:
                snap[g] = getattr(mx, g)()
            except Exception:
                pass
    return snap


def main(argv=None):
    p = argparse.ArgumentParser(prog="mlx-omarchy-laya.qualify")
    p.add_argument("--model", required=True, help="converted checkpoint dir (for identity + direct mode)")
    p.add_argument("--url", default=None, help="qualify a running server instead of a direct engine")
    p.add_argument("--fixture", type=Path, default=FIXTURE)
    p.add_argument("--allow-cpu", action="store_true", help="permit CPU engine (diagnostics only, never qualification)")
    args = p.parse_args(argv)

    fixture = json.loads(args.fixture.read_text())
    repo_root = Path(__file__).resolve().parents[2]
    import mlx.core as mx

    device = str(mx.default_device())
    if device == "cpu" and not args.allow_cpu:
        raise SystemExit("qualify: default device is cpu; run on Apple-silicon hardware with the "
                         "mlx-omarchy wheel (or pass --allow-cpu for a CPU self-check, which is NOT "
                         "a hardware qualification)")

    results = []
    fail = []
    warm_after_first = []
    mem0 = _memory_snapshot()
    for case in fixture["cases"]:
        expect = case["expected_mlx_fp32_cpu"]
        if args.url:
            out, wall_ms = _run_case_url(args.url, case["state"], case["questions"])
        else:
            out, wall_ms = _run_case_direct(args.model, case["state"], case["questions"], not args.allow_cpu)
        prob_max, act_max, n_choice, agree = 0.0, 0.0, 0, 0
        for qid, rq in expect["answers"].items():
            gq = out["answers"][qid]
            if rq["type"] == "choice":
                n_choice += 1
                agree += int(gq["choice"] == rq["choice"])
            if rq["type"] in ("choice", "score"):  # noul answers carry no probabilities block
                for k, v in rq["probabilities"].items():
                    prob_max = max(prob_max, abs(gq["probabilities"][k] - v))
            if rq["type"] == "score":
                prob_max = max(prob_max, abs(gq["score"] - rq["score"]))
            if rq["type"] == "noul":
                prob_max = max(prob_max, abs(gq["noul"] - rq["noul"]))
            act_max = max(act_max, abs(gq["rl_agent"]["act_probability"] - rq["rl_agent"]["act_probability"]))
        tok_ok = out["usage"]["input_tokens"] == expect["usage"]["input_tokens"]
        rate = (agree / n_choice) if n_choice else 1.0  # no choice question -> argmax gate vacuous
        ok = tok_ok and rate >= ARGMAX_GATE and prob_max <= PROB_GATE and act_max <= ACT_GATE
        if not ok:
            fail.append(case["name"])
        results.append({"case": case["name"], "argmax_agreement": rate, "choice_questions": n_choice, "probability_max_abs_error": prob_max,
                        "action_probability_max_abs_error": act_max, "tokens_equal": tok_ok,
                        "forward_ms": out.get("timings", {}).get("predicted_ms"), "wall_ms": round(wall_ms, 2),
                        "verdict": "PASS" if ok else "FAIL"})
        warm_after_first.append(wall_ms)

    print("=" * 72)
    print("laya hardware qualification")
    print("provenance: %s" % _provenance(repo_root))
    print("mlx: %s | device: %s | mode: %s" % (
        mx.__version__, device, "server %s" % args.url if args.url else "direct engine"))
    manifest = Path(args.model) / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        print("checkpoint: %s @ %s (weights sha256 %s)" % (
            m.get("catalog_id"), (m.get("source_revision") or "")[:12],
            (m.get("source_files_sha256") or {}).get("model.safetensors", "?")[:16]))
    print("gates: argmax >= %.0f%%, prob <= %g, act <= %g (fp16 GPU vs fp32 CPU fixture)" % (
        ARGMAX_GATE * 100, PROB_GATE, ACT_GATE))
    for r in results:
        print(json.dumps(r, sort_keys=True))
    cold, warm = warm_after_first[0], warm_after_first[1:]
    print("wall latency: cold %.1f ms, warm %s" % (
        cold, " ".join("%.1f" % w for w in warm) if warm else "n/a (single case)"))
    mem1 = _memory_snapshot()
    print("context/input bounds: max_len %s, head_max_len %s, server question cap 64 per request"
          % (fixture["reference_env"].get("max_len"), fixture["reference_env"].get("head_max_len")))
    print("memory: active %s -> %s, peak %s (bytes; empty dict = this mlx build exposes no counters; "
          "record /proc status on the host as fallback)"
          % (mem0.get("get_active_memory"), mem1.get("get_active_memory"), mem1.get("get_peak_memory")))
    print("VERDICT: %s" % ("PASS" if not fail else "FAIL (%s)" % ", ".join(fail)))
    print("=" * 72)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
