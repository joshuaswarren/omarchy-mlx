"""Interleaved A/B runner with an idle and thermal procedure.

usage: ab.py <out.jsonl> <pairs> <armA> <armB> -- <probe.py> [args...]
Arms are build variants (venv-<arm>, stamp from commits.txt). Each pair
runs A then B, the order alternating per pair (ABBA...). Before every
process: wait until the 1-minute load average is below 0.5 (at most 300 s),
then 20 s idle; record load, the SMC temperature sensors and free memory
before and after. Every process runs under flock /tmp/m1-gpu.lock with a
timeout, and the kernel log is checked for new GPU driver errors after it.
AB_NOWAIT=1 skips the idle wait (correctness runs). Extra environment for both arms: AB_ENV="K=V K2=V2"; per arm:
AB_ENV_A / AB_ENV_B. AB_ROOT overrides the work directory (default
~/gpu/upr); AB_NO_LOCK=1 drops the flock (fixture tests only).

Failure contract: a process that exits non-zero (for example 3, a
provenance refusal, or 124, a timeout) or that adds a GPU driver error to
the kernel log ends the run. Its record is written first, no later arm
runs, and ab.py exits with the child's status (9 for a new kernel error).
GPU error monitoring is required: if `journalctl -k` cannot run or exits
non-zero, ab.py prints a refusal and exits 8 before any arm runs (no
record is written); if it fails after an arm, that arm's record is written
with the monitor error and the run stops with 8. test_ab.py checks these
cases without a GPU.
"""
import json
import os
import subprocess
import sys
import time

U = os.environ.get("AB_ROOT") or os.path.expanduser("~/gpu/upr")
HW = "/sys/class/hwmon/hwmon0"


def commits():
    out = {}
    for line in open(f"{U}/commits.txt"):
        v, c = line.split()
        out[v] = c
    return out


def temps():
    t = {}
    for i in (1, 2, 3, 4):
        try:
            lab = open(f"{HW}/temp{i}_label").read().strip()
            t[lab] = int(open(f"{HW}/temp{i}_input").read()) / 1000
        except OSError:
            pass
    return t


def load1():
    return os.getloadavg()[0]


def state():
    mem = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":")
            if k in ("MemAvailable", "MemTotal"):
                mem[k] = round(int(v.split()[0]) / 2**20, 2)
    except OSError:
        pass
    return {"load1": load1(), "temps_c": temps(), "mem_gib": mem}


class MonitorUnavailable(Exception):
    pass


def gpu_errors():
    try:
        r = subprocess.run(["journalctl", "-k", "-b", "--no-pager", "-q"], capture_output=True, text=True)
    except OSError as e:
        raise MonitorUnavailable(f"journalctl could not run: {e}")
    if r.returncode != 0:
        raise MonitorUnavailable(f"journalctl -k exited {r.returncode}: {r.stderr.strip()[:200]}")
    return sum(1 for l in r.stdout.splitlines()
               if "asahi" in l and any(w in l.lower() for w in ("fault", "error", "timeout", "failed")))


def envfor(arm, extra):
    e = dict(os.environ)
    e.update({
        "VK_ICD_FILENAMES": os.path.expanduser("~/gpu/mesa-install/share/vulkan/icd.d/asahi_icd.aarch64.json"),
        "LD_LIBRARY_PATH": os.path.expanduser("~/gpu/mesa-install/lib"),
        "AGX_APPLE9_MAX_SHARED": "32768", "MLX_OMARCHY_BATCH_WORK": "120000",
        "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "MLX_OMARCHY_NO_MLXLM_WARNING": "1", "TOKENIZERS_PARALLELISM": "false",
        "EXPECT_COMMIT": commits()[arm],
    })
    for kv in (os.environ.get("AB_ENV", "") + " " + extra).split():
        k, v = kv.split("=", 1)
        e[k] = v
    return e


def run_one(arm, probe, extra, timeout=1500):
    t0 = time.time()
    nowait = bool(os.environ.get("AB_NOWAIT"))
    while not nowait and load1() >= 0.5 and time.time() - t0 < 300:
        time.sleep(10)
    if not nowait:
        time.sleep(20)
    before, err0 = state(), gpu_errors()
    lock = [] if os.environ.get("AB_NO_LOCK") else ["flock", "/tmp/m1-gpu.lock"]
    cmd = [*lock, "timeout", str(timeout), f"{U}/venv-{arm}/bin/python", *probe]
    t1 = time.time()
    r = subprocess.run(cmd, cwd=f"{U}/probes", env=envfor(arm, extra), capture_output=True, text=True)
    wall = round(time.time() - t1, 1)
    try:
        new_err, mon_err = gpu_errors() - err0, None
    except MonitorUnavailable as e:
        new_err, mon_err = None, str(e)
    rec = {"arm": arm, "stamp": commits()[arm], "cmd": probe, "env": (os.environ.get("AB_ENV", "") + " " + extra).strip(), "rc": r.returncode,
           "wall_s": wall, "before": before, "after": state(),
           "new_gpu_kernel_errors": new_err, "monitor_error": mon_err,
           "provenance": [l for l in r.stdout.splitlines() if l.startswith("provenance:")],
           "result": [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")],
           "stdout_tail": [l for l in r.stdout.splitlines() if not l.startswith(("{", "provenance:"))][-40:],
           "stderr_tail": r.stderr.splitlines()[-5:]}
    return rec


def main():
    out, pairs, a, b = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    probe = sys.argv[sys.argv.index("--") + 1:]
    c = commits()
    if c[a] == c[b] and os.environ.get("AB_ENV_A", "") == os.environ.get("AB_ENV_B", ""):
        sys.exit(f"REFUSED: both arms stamp {c[a]} with the same environment")
    try:
        gpu_errors()
    except MonitorUnavailable as e:
        print(f"REFUSED: GPU error monitoring is unavailable ({e}); no arm runs", file=sys.stderr, flush=True)
        sys.exit(8)
    with open(out, "a") as f:
        for p in range(pairs):
            order = [(a, "A"), (b, "B")] if p % 2 == 0 else [(b, "B"), (a, "A")]
            for arm, side in order:
                try:
                    rec = run_one(arm, probe, os.environ.get(f"AB_ENV_{side}", ""))
                except MonitorUnavailable as e:
                    print(f"STOP: GPU error monitoring failed before {side} {arm} ({e}); no later arm runs", flush=True)
                    sys.exit(8)
                rec.update({"pair": p, "side": side})
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(side, arm, rec["rc"], json.dumps(rec["result"])[:200], flush=True)
                if rec["monitor_error"]:
                    print(f"STOP: GPU error monitoring failed after {side} {arm} ({rec['monitor_error']}); no later arm runs", flush=True)
                    sys.exit(8)
                if rec["new_gpu_kernel_errors"]:
                    print("STOP: new GPU kernel errors", flush=True)
                    sys.exit(9)
                if rec["rc"] != 0:
                    print(f"STOP: {side} {arm} exited {rec['rc']}; no later arm runs", flush=True)
                    sys.exit(rec["rc"] if 0 < rec["rc"] < 256 else 1)


if __name__ == "__main__":
    main()
