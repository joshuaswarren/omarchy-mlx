#!/usr/bin/env python3
"""Laya head-call probe: per-call NDJSON, resumable, two modes.

phases  in-process decomposition of one system_one call:
        tokenize, collate, arrays, forward-record, mx.eval, readback, post.
        The rebuilt answer must exactly equal a full engine system_one
        call on the same payload (choice, probabilities, act_probability,
        confidence), proving the timed path is the shipped path.
http    wall time against a resident worker through the gate's own
        call_decision/build_payload harness (dev_sweep_run), recording the
        server's prompt_ms/predicted_ms envelope and the residual.

Every timed line carries boot id, uptime, loadavg, PSI, other-process
count, and the arm label. Re-running skips (mode, arm, i) pairs already
present in the NDJSON, so a killed ticket loses nothing.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # gate harness copies (dev_sweep_run)

from dev_sweep_run import QUESTION_DEFAULT, build_payload, call_decision  # noqa: E402


def host_state(server_pid=None):
    load = Path("/proc/loadavg").read_text().strip()
    psi = {}
    for kind in ("some", "full"):
        try:
            line = next(l for l in Path("/proc/pressure/cpu").read_text().splitlines()
                        if l.startswith(kind))
            psi[kind] = dict(kv.split("=") for kv in line.split()[1:])
        except (StopIteration, OSError):
            psi[kind] = None
    out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout
    me = {str(os.getpid()), str(os.getppid())}
    if server_pid:
        me.add(str(server_pid))
    others = 0
    for line in out.splitlines()[1:]:
        parts = line.split(None, 1)
        if len(parts) < 2 or parts[0] in me:
            continue
        exe = Path(parts[1].split()[0]).name  # executable, not full cmdline
        if exe.startswith("python") or exe.startswith("mlx"):
            others += 1
    return {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "uptime_s": float(Path("/proc/uptime").read_text().split()[0]),
        "loadavg": load,
        "psi_cpu": psi,
        "other_python_mlx_procs": others,
        "mlx_env": {k: os.environ.get(k, "unset")
                    for k in ("MLX_OMARCHY_BATCH_WORK", "MLX_OMARCHY_QUEUE_PRIORITY")},
    }


def done_keys(path):
    path = Path(path)
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
                done.add((rec["mode"], rec.get("arm", ""), rec["i"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


class Ndjson:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a")

    def append(self, rec):
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())


def suite_texts(path):
    return [c["text"] for c in json.loads(Path(path).read_text())["cases"]]


def run_phases(args, hs):
    sys.path.insert(0, str(Path(args.serve).resolve()))
    import mlx.core as mx
    import numpy as np
    from mlx_omarchy_laya.api import LayaEngine
    from mlx_omarchy_laya import model as laya_model
    from mlx_omarchy_laya.sequence import (
        collate, confidence_from_probs, encode_questions, temp_bucket)

    engine = LayaEngine(args.model, dtype=mx.float16)

    def fwd(ids, att, mp_, mm, qt):
        return laya_model.forward(engine.weights, engine.enc_cfg, engine.rope,
                                  ids, att, mp_, mm, qt, engine.head_layers)

    if args.compile:
        # Steady-state screen: compile once per shape; timed calls reuse the
        # first text so the loop measures the compiled steady state, and the
        # first call records the compile cost itself.
        fwd = mx.compile(fwd)
    texts = suite_texts(args.suite)
    out = Ndjson(args.out)
    done = done_keys(args.out)
    n = args.n if args.n > 0 else len(texts)  # n = timed calls; texts cycle
    ms = lambda a, z: round((z - a) * 1000.0, 3)

    def one_phased(text):
        """Timed replica of LayaEngine.system_one internals, B=1."""
        questions = build_payload(text, QUESTION_DEFAULT)["questions"]
        t0 = time.perf_counter()
        ids_in_order, items = encode_questions(
            engine.tok, text, questions, engine.max_len, engine.head_max_len)
        t1 = time.perf_counter()
        b = collate(items, engine.tok.pad_token_id)
        t2 = time.perf_counter()
        arrays = (mx.array(b["input_ids"]), mx.array(b["attention_mask"]),
                  mx.array(b["marker_pos"]), mx.array(b["marker_mask"]),
                  mx.array(b["qtype"]))
        t3 = time.perf_counter()
        logits, act = fwd(*arrays)
        t4 = time.perf_counter()
        mx.eval(logits, act)
        t5 = time.perf_counter()
        logits32 = np.array(logits.astype(mx.float32))
        act_p = np.array(mx.softmax(act.astype(mx.float32), axis=-1))
        t6 = time.perf_counter()
        q, k, qt = items[0], len(items[0]["markers"]), items[0]["qtype"]
        z = logits32[0, :k] / engine.temperature_by_options.get(
            temp_bucket(qt, k), engine.temperature[qt])
        p = np.exp(z - z.max())
        p = p / p.sum()
        keys = list(q["crit"].keys())
        answer = {"type": "choice", "choice": keys[int(p.argmax())],
                  "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                  "confidence": round(confidence_from_probs(p, k), 4),
                  "rl_agent": {"act_probability": float(act_p[0, 0])}}
        t7 = time.perf_counter()
        ph = {
            "tokenize_ms": ms(t0, t1), "collate_ms": ms(t1, t2),
            "arrays_ms": ms(t2, t3), "record_ms": ms(t3, t4),
            "eval_ms": ms(t4, t5), "readback_ms": ms(t5, t6),
            "post_ms": ms(t6, t7),
            "seq_len": int(b["input_ids"].shape[1]),
            "n_tokens": int(b["n_tokens"]),
        }
        return ph, answer

    for w in range(args.warmup):  # shader/pipeline warm-up, untimed
        engine.system_one(texts[w % len(texts)], build_payload(
            texts[w % len(texts)], QUESTION_DEFAULT)["questions"])

    mismatches = 0
    for i in range(n):
        if ("phases", args.arm, i) in done:
            continue
        text = texts[0] if args.compile else texts[i % len(texts)]
        rec_hs = host_state()
        t0 = time.perf_counter()
        ph, answer = one_phased(text)
        wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        canon = engine.system_one(text, build_payload(text, QUESTION_DEFAULT)["questions"])
        ref = canon["answers"]["route"]
        equal = (ref == answer)
        mismatches += 0 if equal else 1
        out.append({"mode": "phases", "arm": args.arm, "i": i,
                    "compile": args.compile,
                    "text_sha1": hashlib.sha1(text.encode()).hexdigest()[:12],
                    "wall_ms": wall_ms, "answer_equal": equal,
                    "answer": answer, **ph, "host": rec_hs,
                    "engine_timings": canon.get("timings")})
    if mismatches:
        print(f"FATAL phases arm={args.arm}: {mismatches} answer mismatches", file=sys.stderr)
        sys.exit(3)
    print(f"phases done arm={args.arm} n={n} out={args.out}")


def run_http(args, hs):
    texts = suite_texts(args.suite)
    out = Ndjson(args.out)
    done = done_keys(args.out)
    n = args.n if args.n > 0 else len(texts)  # n = timed calls; texts cycle
    for w in range(args.warmup):
        call_decision(args.url, build_payload(texts[w % len(texts)], QUESTION_DEFAULT),
                      args.timeout)
    for i in range(n):
        if ("http", args.arm, i) in done:
            continue
        text = texts[i % len(texts)]
        hs = host_state(args.server_pid)
        resp, wall_ms, failed = call_decision(args.url,
                                              build_payload(text, QUESTION_DEFAULT),
                                              args.timeout)
        timings = resp.get("timings") or {}
        out.append({
            "mode": "http", "arm": args.arm, "i": i,
            "text_sha1": hashlib.sha1(text.encode()).hexdigest()[:12],
            "wall_ms": round(wall_ms, 3), "failed": failed,
            "server_prompt_ms": timings.get("prompt_ms"),
            "server_predicted_ms": timings.get("predicted_ms"),
            "n_tokens": timings.get("prompt_n"),
            "choice": ((resp.get("answers") or {}).get("route") or {}).get("choice"),
            "host": hs,
        })
    print(f"http done arm={args.arm} n={n} out={args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["phases", "http"])
    ap.add_argument("--serve", default=str(HERE.parent.parent.parent / "serve"),
                    help="dir containing the mlx_omarchy_laya package")
    ap.add_argument("--model", help="Laya checkpoint dir (phases mode)")
    ap.add_argument("--url", help="decision endpoint (http mode)")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--arm", default="")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--compile", action="store_true",
                    help="phases mode: screen an mx.compile'd forward (steady state)")
    ap.add_argument("--server-pid", type=int, default=None)
    args = ap.parse_args()
    if args.mode == "phases":
        run_phases(args, host_state())
    else:
        run_http(args, host_state(args.server_pid))


if __name__ == "__main__":
    main()
