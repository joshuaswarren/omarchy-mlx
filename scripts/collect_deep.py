#!/usr/bin/env python3
"""Collect deep mlx-omarchy diagnostics into one reviewable archive.

For remote performance and correctness work. Self-contained: it needs no
repository build and no model download. Every section detects its own
preconditions, and an unavailable wheel, benchmark binary, or profiling
harness is recorded as data instead of failing the run. Section results
are written as they complete, so a timeout keeps everything that finished.

Sections (schema_version 2):
  quick         the fast capability report (collect_quick.collect)
  environment   python, git commit of the repo, allowlisted env vars
  correctness   fixed-shape mlx ops checked against pure-python references
  benchmark     small matmul timing sweep, plus the kernel spike binary
                when a build is present
  profile       trace smoke, an MLX_OMARCHY_GPU_PROFILE event stream with
                a kernel histogram, and scripts/profile_analyze.py output
  ane           everything needed to turn the ANE on for an untested
                chip (schema v2): macOS IODeviceTree dump of the
                ane / dart-ane / iop-ane / ascwrap / mailbox family with
                all properties (identity keys stripped), ANE driver
                classes and kexts, OS firmware image hashes; Linux
                reserved-memory and mailbox nodes, /proc/iomem ranges,
                firmware file hashes, omarchy-ane-check state, the
                omarchy_ane promotion block, and filtered kernel log.
                An ANE smoke runs only with --ane-smoke and only when
                the tooling exists; it never loads or unloads modules
                and never writes.
  thermal       thermal zone readings before and after the sections

Privacy: every captured value passes through the shared Redactor; raw
logs are capped, never included unredacted. The default run only
previews and uploads nothing. Review the printed manifest, then rerun
with `--out FILE` to write the deterministic archive (sorted members,
fixed mtime, gzip mtime 0: same workspace, same bytes) plus a paste-ready
`FILE.submission.md`. Uploading is always explicit: pass `--submit URL`,
or set MLX_OMARCHY_SUBMIT_URL and type SUBMIT at the prompt. The redacted
summary and archive are sent; the endpoint answers with a public receipt
URL, identical content is deduplicated by its SHA-256, and a failed
upload keeps the local files. The JSON schema does not change with the
sharing path.

"""
import argparse
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import collect_macos
import collect_quick
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
from collect_common import (
    SCHEMA_VERSION,
    Redactor,
    _cap_omarchy_ane,
    archive_bytes,
    build_manifest,
    build_payload,
    is_native_macos,
    json_bytes,
    read_text,
    run_tool,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECTION_TIMEOUTS = {
    "quick": 120,
    "environment": 60,
    "correctness": 300,
    "benchmark": 420,
    "profile": 300,
    "ane": 420,
}
SECTION_ORDER = ("quick", "environment", "correctness", "benchmark",
                 "profile", "ane")

# /proc/interrupts sampling: idle pair 10 s apart, optional third sample
# after the opt-in smoke (a storm shows as rate between the samples).
ANE_INTERRUPT_SAMPLE_SECS = 10
ANE_DMESG_LINES = 200
ANE_DMESG_LINE_BYTES = 160
ANE_DMESG_FAULTS = 32
ANE_DMESG_PATTERN = r"ane_t6021|ane|apple-dart|apple-mailbox|pmgr"
ANE_FAULT_PATTERN = (r"fault|error|timeout|abort|oops|warn|bug|call trace")
# Identity strip list for the raw devicetree text member (same keys as
# the macOS probe strip list).
DT_STRIP_PROPS = re.compile(
    r"(?i)^(serial-number|unique-chip-id|unique-chip|ecid|mlb|"
    r"mac-address|local-mac-address|device-uuid|linux,usable-memory-range"
    r"|wifi-.*|bluetooth-.*|fv-.*|.*-hash)$")
MAX_RAW_DUMP_BYTES = 1024 * 1024
MAX_ADT_DUMP_BYTES = 2 * 1024 * 1024

MAX_STREAM_LINES = 2000

CORRECTNESS_PROBE = r"""
import json, math, time
res = {"available": False, "device": None, "mlx_version": None,
       "import_error": None, "ops": []}
try:
    import mlx.core as mx
except Exception as exc:
    res["import_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(res))
    raise SystemExit(0)
res["available"] = True
res["device"] = str(mx.default_device())
res["mlx_version"] = getattr(mx, "__version__", None)

import sys as _sys
_sys.path.insert(0, __SCRIPTS_DIR__)
from mlx_provenance import prepare_probe

try:
    res["provenance"] = prepare_probe(mx)
    res["device"] = str(mx.default_device())
except Exception as exc:
    res["available"] = False
    res["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(res))
    raise SystemExit(0)
if res["provenance"].get("verified") == "mismatch":
    res["available"] = False
    res["error"] = ("refusing to emit correctness numbers: "
                    + res["provenance"]["mismatch"])
    print(json.dumps(res))
    raise SystemExit(0)

def close(a, b, tol=1e-5):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, list):
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

def record(name, fn, expect):
    entry = {"op": name, "pass": False, "detail": None, "ms": None}
    start = time.perf_counter()
    try:
        got = fn()
        got = got.tolist() if hasattr(got, "tolist") else got
        entry["pass"] = bool(close(got, expect))
        if not entry["pass"]:
            entry["detail"] = f"expected {expect}, got {got}"
    except Exception as exc:
        entry["detail"] = f"{type(exc).__name__}: {exc}"
    entry["ms"] = round((time.perf_counter() - start) * 1000, 3)
    res["ops"].append(entry)

record("add", lambda: mx.array([1.0, 2.0]) + mx.array([3.0, 4.0]), [4.0, 6.0])
record("matmul", lambda: mx.array([[1.0, 2.0], [3.0, 4.0]])
       @ mx.array([[5.0, 6.0], [7.0, 8.0]]),
       [[19.0, 22.0], [43.0, 50.0]])
record("softmax", lambda: mx.softmax(mx.array([1.0, 2.0, 3.0])),
       [0.09003057, 0.24472847, 0.66524096])
record("argmax", lambda: mx.argmax(mx.array([-1.0, 5.0, 3.0])), 1)
record("cumsum", lambda: mx.cumsum(mx.array([1.0, 2.0, 3.0])),
       [1.0, 3.0, 6.0])

xv, wv = [0.5, -0.25], [[1.0, 2.0], [3.0, 4.0]]
xw = [xv[0] * wv[0][j] + xv[1] * wv[1][j] for j in range(2)]
grad_expect = [[2.0 * xw[j] * xv[i] for j in range(2)] for i in range(2)]

def _grad():
    x = mx.array(xv)
    def f(w):
        return mx.sum((x @ w) ** 2)
    return mx.value_and_grad(f)(mx.array(wv))[1]

record("value_and_grad", _grad, grad_expect)
print(json.dumps(res))
"""

BENCH_PROBE = r"""
import json, statistics, time
res = {"available": False, "device": None, "error": None, "matmul": []}
try:
    import mlx.core as mx
    res["available"] = True
    res["device"] = str(mx.default_device())

    import sys as _sys
    _sys.path.insert(0, __SCRIPTS_DIR__)
    from mlx_provenance import prepare_probe

    res["provenance"] = prepare_probe(mx)
    res["device"] = str(mx.default_device())
    if res["provenance"].get("verified") == "mismatch":
        res["available"] = False
        res["error"] = ("refusing to emit timing numbers: "
                        + res["provenance"]["mismatch"])
        print(json.dumps(res))
        raise SystemExit(0)
    sync = getattr(mx, "synchronize", None)
    for n in (256, 512, 1024):
        a = mx.random.normal((n, n))
        b = mx.random.normal((n, n))
        for _ in range(2):
            c = a @ b
            mx.eval(c)
            if sync:
                sync()
        times = []
        for _ in range(8):
            start = time.perf_counter()
            c = a @ b
            mx.eval(c)
            if sync:
                sync()
            times.append((time.perf_counter() - start) * 1000)
        med = statistics.median(times)
        res["matmul"].append({
            "n": n, "reps": len(times),
            "median_ms": round(med, 3),
            "min_ms": round(min(times), 3),
            "max_ms": round(max(times), 3),
            "tflops": round(2 * n ** 3 / (med / 1000) / 1e12, 4),
        })
except Exception as exc:
    res["available"] = False
    res["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(res))
"""

PROFILE_PROBE = r"""
import json, os
out = {"available": False, "error": None}
try:
    import mlx.core as mx
    a = mx.random.normal((256, 256))
    b = mx.random.normal((256, 256))
    for _ in range(3):
        c = a @ b
        s = mx.softmax(a)
        mx.eval(c, s)
    sync = getattr(mx, "synchronize", None)
    if sync:
        sync()
    path = os.environ["MLX_OMARCHY_GPU_PROFILE"]
    out["available"] = os.path.exists(path) and os.path.getsize(path) > 0
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out))
"""


def section_environment(redactor, repo):
    out = {"available": True, "python": platform.python_version(),
           "implementation": platform.python_implementation(),
           "machine": platform.machine(),
           "executable": redactor.apply(sys.executable)}
    out["env"] = [
        {"name": k, "value": redactor.apply(v)[:300]}
        for k, v in sorted(os.environ.items())
        if k.startswith(("MLX_", "MESA_", "VK_"))
    ]
    head = run_tool(["git", "-C", repo, "rev-parse", "HEAD"], redactor,
                    label="git rev-parse HEAD", timeout=15)
    out["source_commit"] = head["stdout"].strip() \
        if head["exit_code"] == 0 else None
    dirty = run_tool(["git", "-C", repo, "status", "--porcelain"], redactor,
                     label="git status count", timeout=15)
    out["repo_dirty"] = dirty["exit_code"] == 0 and bool(dirty["stdout"].strip())
    out["wheels"] = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(repo, "dist", "*.whl")))
    out["prepared_tree"] = os.path.isdir(os.path.join(repo, ".work", "mlx"))
    return out


def section_correctness(redactor):
    rec = run_tool([sys.executable, "-c", CORRECTNESS_PROBE.replace(
        "__SCRIPTS_DIR__", json.dumps(SCRIPTS_DIR))], redactor,
                   label="correctness probe", timeout=240)
    out = {"available": False, "probe": rec, "ops": []}
    if not rec["available"] or rec["exit_code"] != 0:
        return out
    try:
        found = json.loads(rec["stdout"].strip().splitlines()[-1])
    except (ValueError, IndexError):
        out["probe"]["error"] = "unparseable probe output"
        return out
    out["available"] = bool(found.get("available"))
    out["device"] = found.get("device")
    out["mlx_version"] = found.get("mlx_version")
    out["import_error"] = found.get("import_error")
    out["provenance"] = found.get("provenance")
    out["error"] = found.get("error")
    out["ops"] = found.get("ops", [])
    out["ops_passed"] = sum(1 for op in out["ops"] if op.get("pass"))
    return out


def find_spike_binary(repo):
    patterns = (
        ".work/build*/spike/omarchy_matmul_attention",
        ".work/build*/benchmarks/omarchy/omarchy_matmul_attention",
        ".work/build-omarchy/spike/omarchy_matmul_attention",
    )
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(repo, pat))):
            if os.path.isfile(path):
                return path
    return None


def section_benchmark(redactor, repo):
    out = {"available": False, "python": None, "kernel_spike": None}
    rec = run_tool([sys.executable, "-c", BENCH_PROBE.replace(
        "__SCRIPTS_DIR__", json.dumps(SCRIPTS_DIR))], redactor,
                   label="python microbench", timeout=300)
    out["python_probe"] = rec
    if rec["available"] and rec["exit_code"] == 0:
        try:
            found = json.loads(rec["stdout"].strip().splitlines()[-1])
            out["python"] = found
            out["available"] = bool(found.get("available"))
        except (ValueError, IndexError):
            pass
    if platform.system() == "Darwin":
        out["kernel_spike"] = collect_macos.not_applicable()
        return out
    spike = find_spike_binary(repo)
    if spike is None:
        out["kernel_spike"] = {"available": False, "error": "binary not built "
                               "(overlay/benchmarks/omarchy, see its CMakeLists.txt)"}
        return out
    output = os.path.join(tempfile.gettempdir(), "omarchy-spike-result.json")
    argv = [spike, "--output", output, "--warmup", "3", "--reps", "10",
            "--rounds", "1"]
    srec = run_tool(argv, redactor, label="kernel spike", timeout=300)
    out["kernel_spike"] = srec
    out["available"] = out["available"] or srec["exit_code"] == 0
    result = read_text(output)
    if result:
        try:
            out["kernel_spike_result"] = json.loads(result)
        except ValueError:
            out["kernel_spike_result_text"] = result[:20_000]
    return out


def find_info_tool(redactor, repo):
    rec = run_tool([sys.executable, "-c", collect_quick.MLX_PROBE_CODE],
                   redactor, label="info tool locate", timeout=90)
    if rec["exit_code"] == 0:
        try:
            tool = json.loads(rec["stdout"].strip().splitlines()[-1]).get("info_tool")
            if tool:
                return tool
        except (ValueError, IndexError):
            pass
    for pat in (".work/build*/mlx-omarchy-info",
                ".work/build*/tools/mlx-omarchy-info/mlx-omarchy-info"):
        for path in sorted(glob.glob(os.path.join(repo, pat))):
            if os.path.isfile(path):
                return path
    return None


def section_profile(redactor, repo, ws):
    if platform.system() == "Darwin":
        return collect_macos.not_applicable()
    out = {"available": False, "trace_smoke": None, "stream": None,
           "analysis": None}
    tool = find_info_tool(redactor, repo)
    if tool:
        out["trace_smoke"] = run_tool([tool, "--trace-smoke"], redactor,
                                      label="mlx-omarchy-info --trace-smoke",
                                      timeout=60)
        out["available"] = out["trace_smoke"]["exit_code"] == 0
    else:
        out["trace_smoke"] = {"available": False, "error": "not-found",
                              "label": "mlx-omarchy-info --trace-smoke"}
    raw = os.path.join(ws, "raw-stream.jsonl")
    env = dict(os.environ)
    env["MLX_OMARCHY_GPU_PROFILE"] = raw
    probe = run_tool([sys.executable, "-c", PROFILE_PROBE], redactor,
                     label="gpu profile probe", timeout=120, env=env)
    out["profile_probe"] = probe
    if os.path.exists(raw):
        with open(raw, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        hist = {}
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            name = event.get("kernel") or event.get("name") or event.get("k")
            if name:
                hist[name] = hist.get(name, 0) + 1
        kept = lines[:MAX_STREAM_LINES]
        with open(os.path.join(ws, "profile-stream.jsonl"), "w",
                  encoding="utf-8") as fh:
            for line in kept:
                fh.write(redactor.apply(line) + "\n")
        out["stream"] = {
            "available": bool(lines),
            "lines_total": len(lines),
            "lines_included": len(kept),
            "kernel_histogram": sorted(
                hist.items(), key=lambda kv: (-kv[1], kv[0]))[:20],
        }
        out["available"] = out["available"] or bool(lines)
    else:
        out["stream"] = {"available": False,
                         "error": "no event stream; the installed wheel may "
                                  "lack MLX_OMARCHY_GPU_PROFILE tracing"}
    analyze = os.path.join(repo, "scripts", "profile_analyze.py")
    if os.path.exists(analyze) and os.path.exists(raw):
        out["analysis"] = run_tool(
            [sys.executable, analyze, raw], redactor,
            label="profile_analyze.py", timeout=60)
    return out


def _read_int(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _install_token(name, home=None):
    """Per-install random token -> first 16 hex of its sha256.

    The token lives at ~/.config/mlx-omarchy/<name> (mode 0600, created
    on first run). It is random, never a serial/UUID/MAC; the owner may
    copy owner-id to their other machines to link their submissions.
    """
    directory = os.path.join(home or os.path.expanduser("~"),
                             ".config", "mlx-omarchy")
    path = os.path.join(directory, name)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            token = fh.read().strip()
    except OSError:
        token = ""
    if not token:
        token = os.urandom(32).hex()
        try:
            os.makedirs(directory, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(token + "\n")
        except OSError:
            return None
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def _omarchy_ane_check(redactor):
    """`omarchy-ane-check` state: exit, ready/FAILED, UNTESTED, lines."""
    rec = run_tool(["omarchy-ane-check"], redactor,
                   label="omarchy-ane-check", timeout=120)
    if not rec["available"]:
        return {"available": False,
                "unavailable": "omarchy-ane-check not installed"}
    lines = [line[:ANE_DMESG_LINE_BYTES]
             for line in rec["stdout"].splitlines()][:40]
    return {
        "available": True,
        "exit": rec["exit_code"],
        "status": "ready" if rec["exit_code"] == 0 else "FAILED",
        "untested": "untested" in rec["stdout"].lower(),
        "lines": lines,
    }


def _omarchy_ane_module(redactor):
    """The bound ane* module: version, srcversion, params; loaded flag."""
    modules = []
    try:
        names = sorted(d for d in os.listdir("/sys/module")
                       if d.startswith("ane"))
    except OSError:
        names = []
    for name in names[:4]:
        base = os.path.join("/sys/module", name)

        def _prop(prop):
            try:
                with open(os.path.join(base, prop),
                          "r", encoding="utf-8") as fh:
                    return redactor.apply(fh.read().strip()[:128]) or None
            except OSError:
                return None

        params = {}
        try:
            for param in sorted(os.listdir(os.path.join(base,
                                                        "parameters")))[:32]:
                try:
                    with open(os.path.join(base, "parameters", param),
                              "r", encoding="utf-8") as fh:
                        params[param] = redactor.apply(
                            fh.read().strip()[:256])
                except OSError:
                    continue
        except OSError:
            pass
        modules.append({"name": name, "version": _prop("version"),
                        "srcversion": _prop("srcversion"),
                        "params": params or None})
    loaded = None
    try:
        with open("/proc/modules", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("ane"):
                    loaded = redactor.apply(line.strip()[:256])
                    break
    except OSError:
        pass
    module = dict(modules[0]) if modules else {
        "available": False, "unavailable": "no /sys/module/ane* module"}
    return {"module": module, "modules": [m["name"] for m in modules],
            "loaded_line": loaded}


def _omarchy_ane_firmware(redactor):
    """[{path, sha256}] for /lib/firmware/apple/ane/*."""
    root = "/lib/firmware/apple/ane"
    out = {"files": [], "unavailable": None}
    if not os.path.isdir(root):
        out["unavailable"] = "no /lib/firmware/apple/ane directory"
        return out
    for name in sorted(os.listdir(root))[:32]:
        path = os.path.join(root, name)
        if not os.path.isfile(path) or os.path.islink(path):
            continue
        try:
            digest = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            continue
        out["files"].append({
            "path": redactor.apply(path)[:512],
            "sha256": digest.hexdigest(),
        })
    if len(os.listdir(root)) > 32:
        out["truncated"] = "files:max"
    return out


def _omarchy_ane_optin():
    """The ane-* keys of the overlay opt-in file (keys only)."""
    try:
        with open("/etc/omarchy-platform/dtb-overlays.opt-in",
                  "r", encoding="utf-8") as fh:
            return [line.strip()[:128] for line in fh
                    if line.strip().startswith("ane-")][:32]
    except OSError:
        return []


# The packaged ANE smoke runner (omarchy-ane). w73 ships the final
# command name; override with MLX_OMARCHY_ANE_SMOKE_RUNNER until then.
ANE_SMOKE_RUNNER = "omarchy-ane-smoke"


def _omarchy_ane_smoke(redactor):
    """Opt-in ANE smoke: the packaged add-fixture runner.

    Runs the shipped add ANEC 20 times. Exit 0 = all bit-exact; exit 1
    = ran but some calls failed (parse the JSON anyway and keep it:
    available true with errors > 0 is exactly what the promotion check
    must see); exit 2 = unavailable for this SoC. stdout is exactly one
    JSON line; stderr one "omarchy-ane-smoke: ..." line used as the
    reason. Consent is checked by the caller; bounded timeout; never
    loads or unloads modules, never writes.
    """
    runner_name = os.environ.get("MLX_OMARCHY_ANE_SMOKE_RUNNER",
                                 ANE_SMOKE_RUNNER)
    smoke = {"available": False, "name": "add-fixture",
             "chip": None, "sha256": [], "golden_sha256": None,
             "errors": 0, "min_ms": None, "median_ms": None,
             "exit": None,
             "reason": f"smoke runner '{runner_name}' not shipped in "
                       "omarchy-ane yet"}
    runner = shutil.which(runner_name)
    if runner is None:
        return smoke
    rec = run_tool([runner_name], redactor,
                   label=runner_name, timeout=120)
    smoke["exit"] = rec["exit_code"]
    stderr_line = rec["stderr"].strip().splitlines()[-1] \
        if rec["stderr"].strip() else ""
    if rec["exit_code"] in (0, 1):
        try:
            parsed = json.loads(rec["stdout"].strip().splitlines()[-1])
        except (ValueError, IndexError):
            smoke["reason"] = stderr_line or \
                "runner printed no JSON summary"
            return smoke
        for key in ("name", "chip", "available", "sha256",
                    "golden_sha256", "errors", "min_ms", "median_ms",
                    "reason"):
            if key in parsed:
                smoke[key] = parsed[key]
        smoke["available"] = bool(parsed.get("available",
                                             rec["exit_code"] == 0))
        smoke["reason"] = (str(parsed.get("reason"))
                           [:256] if parsed.get("reason")
                           else (stderr_line[:256]
                                 if rec["exit_code"] == 1 else None))
        return smoke
    if rec["exit_code"] == 2:
        smoke["reason"] = stderr_line or rec["stdout"][:256] or \
            "runner reported unavailable"
        return smoke
    smoke["reason"] = stderr_line or "exit %s" % rec["exit_code"]
    return smoke


def _ane_kernel_log(redactor):
    """Filtered kernel log: first N matching lines + fault subset.

    journalctl -k -b is preferred (boot-scoped, chronological, so the
    FIRST lines of the boot are kept); dmesg is the fallback.
    """
    source = "journalctl"
    rec = run_tool(
        ["sh", "-c",
         "journalctl -k -b --no-pager -q 2>/dev/null | "
         f"grep -iE '{ANE_DMESG_PATTERN}' || true"],
        redactor, label="journalctl ane filter", timeout=30)
    if rec["exit_code"] != 0 or not rec["stdout"].strip():
        source = "dmesg"
        rec = run_tool(
            ["sh", "-c",
             f"dmesg 2>/dev/null | grep -iE '{ANE_DMESG_PATTERN}' || true"],
            redactor, label="dmesg ane filter", timeout=30)
    lines = [line[:ANE_DMESG_LINE_BYTES]
             for line in rec["stdout"].splitlines()]
    fault_re = re.compile(ANE_FAULT_PATTERN, re.I)
    # Faults are drawn from the FULL filtered stream, not only the kept
    # first window: a late-boot fault is exactly what promotion needs.
    faults = [line[:ANE_DMESG_LINE_BYTES]
              for line in rec["stdout"].splitlines()
              if fault_re.search(line)]
    return {"source": source if lines else None,
            "dmesg": lines[:ANE_DMESG_LINES],
            "dmesg_faults": faults[:ANE_DMESG_FAULTS]}


def _ane_interrupts_sample():
    """ANE/DART/mailbox lines of /proc/interrupts (one sample)."""
    try:
        with open("/proc/interrupts", "r", encoding="utf-8",
                  errors="replace") as fh:
            return [line.rstrip("\n")[:512] for line in fh
                    if re.search(r"ane|dart|mailbox", line, re.I)][:64]
    except OSError:
        return None


def _ane_interrupts(redactor, smoke_ran):
    """Idle pair 10 s apart; a third sample after the smoke when it ran."""
    phases = [("idle_first", 0), ("idle_second", ANE_INTERRUPT_SAMPLE_SECS)]
    if smoke_ran:
        phases.append(("after_smoke", ANE_INTERRUPT_SAMPLE_SECS))
    out = []
    for phase, pause in phases:
        if pause:
            time.sleep(pause)
        out.append({"phase": phase, "lines": _ane_interrupts_sample()})
    return out


def _ane_linux_dt_text(redactor, ws):
    """Archive member: dtc text of the booted tree, stripped, capped.

    Uses dtc when present; a missing dtc is recorded, not fatal.
    Identity props are dropped from the TEXT so the raw member obeys
    the same strip list as everything else.
    """
    rec = run_tool(["dtc", "-q", "-I", "fs", "-O", "dts",
                    collect_quick.DT_BASE], redactor,
                   label="dtc devicetree text", timeout=60)
    if rec["exit_code"] != 0 or not rec["stdout"].strip():
        return None, rec["error"] or "dtc unavailable"
    kept = []
    for line in rec["stdout"].splitlines():
        stripped = line.strip()
        if "=" in stripped:
            label = stripped.split("=", 1)[0].strip().strip('";')
            if DT_STRIP_PROPS.match(label):
                kept.append(f"\t{label}; // [stripped]")
                continue
        kept.append(redactor.apply(line[:512]))
    truncated = False
    if len("\n".join(kept)) > MAX_RAW_DUMP_BYTES:
        truncated = True
        text = "\n".join(kept)[:MAX_RAW_DUMP_BYTES]
    else:
        text = "\n".join(kept)
    with open(os.path.join(ws, "ane-linux-dt.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
        if truncated:
            fh.write("[truncated]\n")
    return True, None


def _generation_probe(nodes):
    """Per-pattern hit counts over captured node names (E.4 shape)."""
    patterns = ("ane", "iop-ane", "ascwrap", "exclave", "sk-", "sio-ane",
                "dart-ane", "ane-dart", "t8020")
    probe = []
    for pattern in patterns:
        rx = re.compile(re.escape(pattern), re.I)
        matched = 0
        for node in nodes or []:
            hay = " ".join(str(node.get(k) or "") for k in
                           ("name", "path", "compatible"))
            if rx.search(hay):
                matched += 1
        probe.append({"pattern": pattern, "matched": matched})
    return probe


def section_ane(redactor, repo, ws, smoke):
    """Everything needed to turn the ANE on for an untested chip.

    macOS: the full IODeviceTree dump of the ANE family (pattern-based,
    so M3 ascwrap IOPs and M4 t8020-class nodes match without code
    changes), ANE driver classes, loaded kexts, OS firmware image
    hashes (names/sizes/sha256 of public paths, never the files), and
    the optional CoreML smoke. Linux: the omarchy_ane promotion block
    (install ids, check, module, firmware, opt-in keys, smoke, kernel
    log), /proc/interrupts samples, package versions, host facts, the
    full reserved-memory subtree, and a dtc text dump as an archive
    member. Every capture that cannot be read says `unavailable`
    (with the reason) instead of being omitted.
    """
    out = {"available": False, "platform": platform.system()}
    if platform.system() == "Darwin":
        dump = collect_macos.probe_ane_dump(redactor)
        detail = dump.get("dump") or {}
        out["generation_probe"] = _generation_probe(detail.get("nodes"))
        raw_text = detail.pop("raw_text", None)
        if raw_text:
            with open(os.path.join(ws, "ane-macos-iodt.txt"), "w",
                      encoding="utf-8") as fh:
                fh.write(raw_text)
        out["dump"] = dump
        out["smoke"] = (collect_macos.probe_ane_smoke(redactor)
                        if smoke else {"requested": False})
        out["os"] = {
            "product_version": _text_of(run_tool(
                ["sw_vers", "-productVersion"], redactor, timeout=10)),
            "build": _text_of(run_tool(
                ["sw_vers", "-buildVersion"], redactor, timeout=10)),
        }
        out["hardware"] = {
            "model_identifier": _text_of(run_tool(
                ["sysctl", "-n", "hw.model"], redactor, timeout=10)),
            "machine": _text_of(run_tool(
                ["uname", "-m"], redactor, timeout=10)),
            "ncpu": _int_of(_text_of(run_tool(
                ["sysctl", "-n", "hw.ncpu"], redactor, timeout=10))),
            "memsize_bytes": _int_of(_text_of(run_tool(
                ["sysctl", "-n", "hw.memsize"], redactor, timeout=10))),
        }
        out["kext_facts"] = {
            "extensions_ane": _ls_grep(redactor,
                                       "/System/Library/Extensions", "ane"),
            "framework_present": os.path.isdir(
                "/System/Library/PrivateFrameworks/AppleNeuralEngine."
                "framework"),
        }
        out["available"] = bool((dump.get("dump") or {}).get("available"))
        return out
    # Linux.
    out["machine_id"] = _install_token("machine-id")
    out["owner_id"] = _install_token("owner-id")
    out["check"] = _omarchy_ane_check(redactor)
    module_state = _omarchy_ane_module(redactor)
    out["module"] = module_state["module"]
    out["modules"] = module_state["modules"]
    out["loaded_line"] = module_state["loaded_line"]
    out["firmware"] = _omarchy_ane_firmware(redactor)
    out["opt_in"] = _omarchy_ane_optin()
    out["uptime_s"] = _uptime_s()
    out["boot_id"] = _boot_id()
    smoke_result = {"requested": False}
    if smoke:
        if out["check"].get("status") == "ready":
            smoke_result = _omarchy_ane_smoke(redactor)
        else:
            smoke_result = {"requested": True, "available": False,
                            "reason": "omarchy-ane-check is not ready; "
                                      "the smoke needs omarchy-ane "
                                      "installed and the module bound"}
    out["smoke"] = smoke_result
    kernel_log = _ane_kernel_log(redactor)
    out["dmesg"] = kernel_log["dmesg"]
    out["dmesg_faults"] = kernel_log["dmesg_faults"]
    out["dmesg_source"] = kernel_log["source"]
    out["iomem"] = _ane_iomem_ranges(redactor)
    out["reserved_memory"] = collect_quick._ane_reserved_memory(
        redactor, max_nodes=32)
    out["interrupts"] = _ane_interrupts(redactor, smoke_result.get(
        "available") is True)
    out["packages"] = _ane_packages(redactor)
    out["host"] = {
        "cpu_online": _cpu_online(),
        "mem_total_mib": collect_quick._mem_total(),
        "lscpu_model": _lscpu_model(redactor),
        "uptime_s": out["uptime_s"],
    }
    dt_dumped, dt_error = _ane_linux_dt_text(redactor, ws)
    if not dt_dumped:
        out["dt_text"] = {"available": False, "unavailable": dt_error}
    else:
        out["dt_text"] = {"available": True}
    out["generation_probe"] = _generation_probe(
        [{"name": name} for name in _linux_ane_node_names()])
    out["available"] = True
    return out


def _linux_ane_node_names():
    """Node names under the booted DT matching the ANE family patterns."""
    import collect_quick
    names = []
    base = collect_quick.DT_BASE
    rx = re.compile(r"ane|ascwrap|exclave", re.I)
    for dirpath, dirs, _files in os.walk(base):
        dirs.sort()
        base_name = os.path.basename(dirpath)
        if rx.search(base_name):
            names.append({"name": os.path.relpath(dirpath, base)[:256]
                          if dirpath != base else "/"})
        if len(names) >= 24:
            break
    return names


def _text_of(rec):
    return rec["stdout"].strip()[:256] if rec["exit_code"] == 0 else None


def _int_of(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _ls_grep(redactor, directory, needle):
    rec = run_tool(["sh", "-c", f"ls {directory} 2>/dev/null | grep -i "
                                f"{needle} || true"],
                   redactor, label=f"ls {directory}", timeout=15)
    if rec["exit_code"] != 0:
        return {"unavailable": "ls failed"}
    return [line[:256] for line in rec["stdout"].splitlines()][:16]


def _uptime_s():
    try:
        with open("/proc/uptime", "r", encoding="utf-8") as fh:
            return int(float(fh.read().split()[0]))
    except (OSError, ValueError, IndexError):
        return None


def _boot_id():
    try:
        with open("/proc/sys/kernel/random/boot_id", "r",
                  encoding="utf-8") as fh:
            return hashlib.sha256(fh.read().strip().encode()).hexdigest()[:12]
    except OSError:
        return None


def _cpu_online():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def _lscpu_model(redactor):
    rec = run_tool(["lscpu"], redactor, label="lscpu", timeout=15)
    if rec["exit_code"] != 0:
        return None
    for line in rec["stdout"].splitlines():
        if line.lower().startswith("model name:"):
            return redactor.apply(line.split(":", 1)[1].strip()[:256])
    return None


def _ane_packages(redactor):
    rec = run_tool(["pacman", "-Q"], redactor, label="pacman -Q",
                   timeout=30)
    if rec["exit_code"] != 0:
        return {"unavailable": "pacman not available"}
    wanted = re.compile(r"^(omarchy-ane|linux-asahi|m1n1|uboot-asahi)")
    return [{"name": parts[0], "version": parts[1]}
            for parts in (line.split()[:2] for line in
                          rec["stdout"].splitlines())
            if len(parts) == 2 and wanted.match(parts[0])][:16]


def _ane_iomem_ranges(redactor):
    """/proc/iomem ranges relevant to the ANE (ane/dart/pmgr/reserved)."""
    try:
        with open("/proc/iomem", "r", encoding="utf-8",
                  errors="replace") as fh:
            return [redactor.apply(line.rstrip("\n")[:512]) for line in fh
                    if re.search(r"ane|dart|pmgr|reserved", line, re.I)][:64]
    except OSError:
        return None


def read_thermal(redactor):
    zones = []
    base = "/sys/class/thermal"
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return zones
    for zone in entries:
        if not zone.startswith("thermal_zone"):
            continue
        temp = read_text(os.path.join(base, zone, "temp"))
        if not temp:
            continue
        kind = read_text(os.path.join(base, zone, "type")) or ""
        try:
            value = int(temp.strip())
        except ValueError:
            continue
        zones.append({"zone": zone, "type": redactor.apply(kind.strip()),
                      "temp_mc": value})
    return zones


def run_sections(ws, repo, redactor, timeout_override, skip, smoke=False):
    """Run each section as a bounded child; keep whatever completed."""
    for name in SECTION_ORDER:
        if name in skip:
            continue
        timeout = timeout_override or SECTION_TIMEOUTS[name]
        argv = [sys.executable, os.path.abspath(__file__), "--_section",
                name, "--_workspace", ws, "--_repo", repo]
        if smoke:
            argv.append("--_ane_smoke")
        rec = run_tool(argv, redactor, label=f"section:{name}",
                       timeout=timeout)
        if not os.path.exists(os.path.join(ws, f"{name}.json")):
            with open(os.path.join(ws, f"{name}.json"), "wb") as fh:
                fh.write(json_bytes({
                    "available": False,
                    "error": rec["error"] or f"exit {rec['exit_code']}",
                    "stderr": rec["stderr"][-2000:],
                }))


def assemble_files(ws, repo, thermal):
    """Member dict from whatever the sections produced; partial is fine."""
    files = {}
    unavailable = []
    redaction = {}
    for name in SECTION_ORDER:
        path = os.path.join(ws, f"{name}.json")
        data = read_text(path)
        if data is None:
            data = json_bytes({"available": False,
                               "error": "section produced no result"})
            unavailable.append(name)
        else:
            try:
                parsed = json.loads(data)
                if parsed.get("available") is False:
                    unavailable.append(name)
                for kind, count in parsed.get("_redaction", {}).items():
                    redaction[kind] = redaction.get(kind, 0) + count
            except ValueError:
                pass
        files[f"{name}.json"] = data if isinstance(data, bytes) \
            else data.encode("utf-8")
    stream = os.path.join(ws, "profile-stream.jsonl")
    if os.path.exists(stream):
        with open(stream, "rb") as fh:
            files["profile-stream.jsonl"] = fh.read()
    for member in ("ane-linux-dt.txt", "ane-macos-iodt.txt",
                   "ane-adt-dump.bin"):
        path = os.path.join(ws, member)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                files[member] = fh.read()
    files["thermal.json"] = json_bytes({"zones": thermal})
    if platform.system() == "Darwin":
        files["thermal.json"] = json_bytes({
            "available": False, "zones": [],
            "error": "macOS temperature sensors are not collected",
        })
        unavailable.append("thermal")
    return files, unavailable, redaction


SUBMISSION_MEMBERS = ("manifest.json", "submission.md")


def _member(files, name):
    try:
        return json.loads(files[name].decode("utf-8"))
    except (KeyError, ValueError):
        return {}


def _quick_with_probe_metadata(files, manifest):
    quick = _member(files, "quick.json")
    if not is_native_macos(quick.get("host") or {}, manifest):
        return quick
    mlx = dict(quick.get("mlx") or {})
    probes = (_member(files, "correctness.json"),
              _member(files, "benchmark.json").get("python") or {})
    for probe in probes:
        if not probe.get("available"):
            continue
        provenance = probe.get("provenance") or {}
        mlx["mlx_version"] = (mlx.get("mlx_version") or probe.get("mlx_version")
                              or provenance.get("mx_version") or provenance.get("dist_version"))
        mlx["default_device"] = mlx.get("default_device") or probe.get("device")
        if mlx.get("metal_available") is None:
            # Native probes require Metal before reporting availability.
            mlx["metal_available"] = True
    return {**quick, "mlx": mlx}


def build_submission(manifest, files, archive_name):
    """Paste-ready cover text from the manifest and collected sections."""
    quick = _quick_with_probe_metadata(files, manifest)
    host = quick.get("host", {})
    mesa = quick.get("mesa", {}).get("gpu", {})
    mlx = quick.get("mlx", {})
    correctness = _member(files, "correctness.json")
    ops = correctness.get("ops", [])
    lines = ["## mlx-omarchy hardware report", ""]
    if is_native_macos(host, manifest):
        metal = mlx.get("metal_available")
        lines += [
            f"Machine: {host.get('model') or 'unknown'} / {host.get('chip') or 'unknown'}"
            f" ({host.get('os') or 'macOS'}, Darwin {host.get('kernel_release') or 'unknown'})",
            f"Native MLX: {mlx.get('mlx_version') or 'unknown'}, "
            f"Metal available: {metal if metal is not None else 'unknown'}",
            "Native macOS reference only. This does not prove Linux support, "
            "ANE execution, or performance parity.",
        ]
    else:
        lines += [
            f"Machine: {host.get('devicetree', {}).get('model') or 'unknown model'}"
            f" ({host.get('arch', 'unknown arch')}, kernel "
            f"{host.get('kernel_release', 'unknown')})",
            f"Vulkan: {mesa.get('deviceName') or 'unavailable'} / "
            f"{mesa.get('driverName') or 'unavailable'}, API "
            f"{mesa.get('apiVersion') or 'unavailable'}",
            f"mlx-omarchy: {mlx.get('distributions', {}).get('mlx-omarchy') or 'not installed'}"
            f", device {mlx.get('default_device') or 'unavailable'}",
        ]
    lines += [
        f"Source commit: {manifest.get('source_commit') or 'unknown'}",
        f"Correctness: {sum(1 for op in ops if op.get('pass'))}/{len(ops)}"
        f" probe ops pass"
        + (" (probes unavailable; see correctness.json)"
           if correctness.get("available") is False else ""),
        f"Not available on this machine: "
        f"{', '.join(manifest.get('sections_unavailable', [])) or 'nothing'}",
        f"Redaction applied before writing: "
        f"{json.dumps(manifest.get('redaction_summary', {}), sort_keys=True)}"
        " (names, paths, IPs, MACs, serials, credentials; no upload code)",
        "",
        "Members with SHA-256 (see attached archive for full contents):",
        "",
        "```",
    ]
    for entry in manifest.get("files", []):
        lines.append(f"{entry['sha256']}  {entry['path']} ({entry['bytes']} B)")
    lines += ["```", "",
              "<details><summary>quick report JSON</summary>", "",
              "```json",
              files.get("quick.json", b"{}").decode("utf-8").strip(),
              "```", "", "</details>", ""]
    return "\n".join(lines)


def _ane_result(files):
    """Parsed ane.json section data, or None."""
    try:
        data = json.loads(files["ane.json"].decode("utf-8"))
    except (KeyError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def finalize(files, unavailable, redaction, archive_name, repo):
    commit = None
    dirty = None
    head = run_tool(["git", "-C", repo, "rev-parse", "HEAD"], Redactor(),
                    label="git rev-parse HEAD", timeout=15)
    if head["exit_code"] == 0:
        commit = head["stdout"].strip()
        dirty_rec = run_tool(["git", "-C", repo, "status", "--porcelain"],
                             Redactor(), label="git status count", timeout=15)
        dirty = bool(dirty_rec["stdout"].strip())
    listed = {name: data for name, data in files.items()
              if name not in SUBMISSION_MEMBERS}
    manifest = build_manifest(archive_name, listed, extra={
        "source_commit": commit,
        "repo_dirty": dirty,
        "system": platform.system(),
        "sections_unavailable": unavailable,
        "redaction_summary": dict(sorted(redaction.items())),
        "schema_note": "one file per section; probe records carry argv, "
                       "exit code, capped redacted output; the schema does "
                       "not change with the sharing path",
    })
    quick = _quick_with_probe_metadata(files, manifest)
    bench = json.loads(files.get("benchmark.json", b"{}").decode("utf-8")
                       or "{}")
    matmul = ((bench.get("python") or {}).get("matmul") or []) \
        if isinstance(bench, dict) else []
    ane_result = _ane_result(files)
    ane_macos = ane_linux = None
    if ane_result is not None:
        redactor = Redactor()
        if ane_result.get("platform") == "Darwin":
            ane_macos = ane_result
        else:
            ane_linux = ane_result
            # The promotion contract path: ane_port_detail.runtime
            # .omarchy_ane. Inject the capped block into the quick view
            # so build_payload carries it there.
            if quick.get("ane_port") is None:
                quick["ane_port"] = {}
            quick["ane_port"].setdefault("runtime", {})[
                "omarchy_ane"] = _cap_omarchy_ane(ane_result, redactor)
    payload = build_payload("deep", quick, manifest, benchmark=matmul,
                            redactor=Redactor(), ane_macos=ane_macos,
                            ane_linux=ane_linux)
    files["manifest.json"] = json_bytes(manifest)
    files["submission.md"] = build_submission(
        manifest, files, archive_name).encode("utf-8")
    return manifest, archive_bytes(files), payload


def print_preview(manifest, data, archive_name, will_write):
    print(dump_preview(manifest), end="")
    print(f"[preview] archive: {archive_name} bytes={len(data)} "
          f"sha256={hashlib.sha256(data).hexdigest()}")
    if not will_write:
        print("[preview] nothing written, nothing uploaded; "
              "rerun with --out FILE to write these exact bytes")


def dump_preview(manifest):
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def section_child(name, ws, repo, smoke=False):
    redactor = Redactor()
    try:
        context = None
        if name == "benchmark" and platform.system() == "Darwin":
            context = collect_macos.measurement_context()
        if name == "quick":
            data = collect_quick.collect()
        elif name == "environment":
            data = section_environment(redactor, repo)
        elif name == "correctness":
            data = section_correctness(redactor)
        elif name == "benchmark":
            data = section_benchmark(redactor, repo)
        elif name == "profile":
            data = section_profile(redactor, repo, ws)
        elif name == "ane":
            data = section_ane(redactor, repo, ws, smoke)
        else:
            data = {"available": False, "error": f"unknown section {name}"}
        if context is not None:
            data["context"] = {"before": context,
                               "after": collect_macos.measurement_context()}
    except Exception as exc:
        data = {"available": False,
                "error": redactor.apply(f"{type(exc).__name__}: {exc}")}
    data = redactor.apply_value(data)
    if isinstance(data, dict):
        data["_redaction"] = redactor.counts
    with open(os.path.join(ws, f"{name}.json"), "wb") as fh:
        fh.write(json_bytes(data))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", metavar="FILE",
                    help="write the archive after the preview (default: "
                         "preview only)")
    ap.add_argument("--submit", metavar="URL", default=None,
                    help="upload the redacted archive to this endpoint "
                         "after the preview")
    ap.add_argument("--repo", default=REPO,
                    help="repository root (default: parent of scripts/)")
    ap.add_argument("--workspace", metavar="DIR",
                    help="keep the section files in DIR instead of a temp dir")
    ap.add_argument("--skip", default="",
                    help="comma-separated sections to skip")
    ap.add_argument("--timeout", type=int, default=None,
                    help="override the per-section timeout in seconds")
    ap.add_argument("--ane-smoke", action="store_true",
                    help="opt-in ANE smoke (Linux: the packaged "
                         "add-fixture runner when omarchy-ane ships it; "
                         "it; macOS: a tiny CoreML add model). Never "
                         "loads or unloads modules, never writes.")
    ap.add_argument("--adt-dump", metavar="FILE", default=None,
                    help="attach an m1n1 ADT dump (developer run, capped "
                         "at 2 MiB; the developer is responsible for "
                         "stripping identity properties)")
    ap.add_argument("--_section", help=argparse.SUPPRESS)
    ap.add_argument("--_workspace", help=argparse.SUPPRESS)
    ap.add_argument("--_repo", help=argparse.SUPPRESS)
    ap.add_argument("--_ane_smoke", action="store_true",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._section:
        section_child(args._section, args._workspace, args._repo,
                      smoke=args._ane_smoke)
        return

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    keep = False
    if args.workspace:
        os.makedirs(args.workspace, exist_ok=True)
        ws = args.workspace
        keep = True
    else:
        ws = tempfile.mkdtemp(prefix="mlx-omarchy-deep-")

    pre_redactor = Redactor()
    thermal_start = read_thermal(pre_redactor)
    run_sections(ws, os.path.abspath(args.repo), pre_redactor,
                 args.timeout, skip, smoke=args.ane_smoke)
    thermal_end = read_thermal(pre_redactor)
    thermal = [{"phase": "start", **z} for z in thermal_start] + \
              [{"phase": "end", **z} for z in thermal_end]
    if args.adt_dump:
        try:
            with open(args.adt_dump, "rb") as fh:
                blob = fh.read(MAX_ADT_DUMP_BYTES + 1)
            if len(blob) > MAX_ADT_DUMP_BYTES:
                print(f"[adt] {args.adt_dump} exceeds "
                      f"{MAX_ADT_DUMP_BYTES} bytes; not attached",
                      file=sys.stderr)
            else:
                with open(os.path.join(ws, "ane-adt-dump.bin"), "wb") as fh:
                    fh.write(blob)
                print(f"[adt] attached {len(blob)} bytes as "
                      f"ane-adt-dump.bin (sha256="
                      f"{hashlib.sha256(blob).hexdigest()[:16]}...)")
        except OSError as exc:
            print(f"[adt] unreadable: {exc}", file=sys.stderr)

    files, unavailable, redaction = assemble_files(
        ws, os.path.abspath(args.repo), thermal)
    for kind, count in pre_redactor.counts.items():
        redaction[kind] = redaction.get(kind, 0) + count
    archive_name = os.path.basename(args.out) if args.out \
        else "mlx-omarchy-deep.tar.gz"
    manifest, data, payload = finalize(files, unavailable, redaction,
                                       archive_name, os.path.abspath(args.repo))
    print_preview(manifest, data, archive_name, bool(args.out))
    if args.out:
        with open(args.out, "wb") as fh:
            fh.write(data)
        base = args.out.removesuffix(".tar.gz") \
            if args.out.endswith(".tar.gz") \
            else os.path.splitext(args.out)[0]
        submission = base + ".submission.md"
        with open(submission, "wb") as fh:
            fh.write(files["submission.md"])
        print(f"[receipt] wrote {args.out} ({len(data)} bytes, "
              f"sha256={hashlib.sha256(data).hexdigest()})")
        print(f"[receipt] wrote {submission} (paste-ready cover text)")
    exit_code = maybe_submit(args, data, archive_name, args.out, payload)
    if not keep:
        shutil.rmtree(ws, ignore_errors=True)
    raise SystemExit(exit_code)


def maybe_submit(args, data, archive_name, out_path, payload):
    """Upload only after explicit consent; never without a local copy."""
    import collect_submit
    endpoint = collect_submit.endpoint_from_args(args)
    if endpoint is None:
        if out_path:
            print(f"[receipt] done; nothing was uploaded. To share it, rerun with "
                  f"--submit {collect_submit.DEFAULT_ENDPOINT} or send {out_path} "
                  f"and its .submission.md by hand.")
        return 0
    if not out_path:
        print("[submit] no --out file; refusing to upload without keeping "
              "a local copy", file=sys.stderr)
        return 4
    digest = hashlib.sha256(data).hexdigest()
    if not args.submit and not collect_submit.confirm_interactive(
            endpoint, archive_name, digest):
        print(f"[submit] declined; {out_path} stays local")
        return 0
    try:
        receipt = collect_submit.submit(endpoint, data, payload,
                                        token=collect_submit.token_from_env())
    except collect_submit.SubmitError as exc:
        print(f"[submit] FAILED: {exc}", file=sys.stderr)
        print(f"[submit] local output preserved: {out_path}", file=sys.stderr)
        return 4
    print(f"[receipt] public URL: {receipt['url']} "
          f"(deduplicated={receipt['deduplicated']})")
    return 0


if __name__ == "__main__":
    main()
