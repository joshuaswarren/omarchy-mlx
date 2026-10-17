#!/usr/bin/env python3
"""Workload-matrix runner for mlx-omarchy / native-MLX baseline runs.

Drives the existing scripts/bench_decode.py over the matrix declared in
scripts/bench_matrix.json: every non-skipped model x every workload, on
Linux (mlx-omarchy) and macOS (upstream mlx) alike. The runner adds
nothing to the metrics: it resolves model snapshots, checks the
environment, spawns the benchmark CLI, and parses the lines it prints.

Rules this runner enforces:
  - A model whose pinned revision is not in the local Hugging Face cache
    is reported skipped. It is never fetched and never passes.
  - Optional models with revision null get their factual revision from
    the local cache snapshot directory at run time. Absent means skipped.
    Revisions are never guessed.
  - Only a leg whose benchmark process exited 0 and printed a decode
    rate has status "measured". Statuses are measured | skipped | failed.
  - Every run records host provenance (chip, cores, memory, OS, GPU
    driver identity), power state, and whether other model-serving
    processes are running. Contended runs are labeled; they are never
    silently mixed with clean numbers.
  - Recorded output carries no hostname, user name, serial, or absolute
    home paths.

Modes:
  plan       resolve the matrix and models, capture provenance, run
             nothing (default)
  metadata   provenance, power, and clean-check only
  run        execute every ready leg through scripts/bench_decode.py

Exit codes: 0 ok (or at least one measured leg), 1 bad invocation or
manifest, 3 run mode finished with zero measured legs, 4 --expect-pins
refusal: a resolved revision differs from the expected pin, so the run
is not comparable and nothing executes.

Self-test (no GPU, no mlx): python3 bench_matrix.py --self-test
"""

import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "bench-matrix/1"
SCRIPTS_DIR = Path(__file__).resolve().parent

# Other model-serving processes that make timing contended. Matched
# against process names only; arguments are never recorded.
CONTENDED_NAMES = (
    "llama-server", "llama-cli", "llama-cpp", "ollama", "omlx",
    "docling", "ane-qwen-model",
)

DECODE_RE = re.compile(
    r"decode ([0-9.]+) tok/s over (\d+) tokens \((\d+) requested")
PREFILL_RE = re.compile(r"prefill ([0-9.]+)s")
PER_TOKEN_RE = re.compile(r"decode mean per-token ([0-9.]+) ms")
PROV_LINE_RE = re.compile(r"provenance: (.+)")
IDS_RE = re.compile(r"generated_ids sha256:([0-9a-f]+) n=(\d+)")
PROMPT_TOKENS_RE = re.compile(r"prompt_tokens (\d+)")
# bench_decode.py prints "uclamp: <outcome>" before the provenance line.
UCLAMP_RE = re.compile(r"^uclamp: (.+)$")


def sanitize(text, home=None):
    """Strip personal identifiers: home paths and nothing else."""
    home = home or os.path.expanduser("~")
    return str(text).replace(home, "~")


# ---------------------------------------------------------------- models

def model_repo_dir(hf_cache, repo):
    return Path(hf_cache) / ("models--" + repo.replace("/", "--"))


def resolve_model(entry, hf_cache, overrides):
    """Return (status, model_path_or_None, revision, reason, rev_class).

    status is "ready" or "skipped". revision is the factual revision
    only when it is backed by trusted cache metadata (a snapshot
    directory named by its commit SHA inside the HF cache). A --model-dir
    override proves nothing about its weights: it reports revision null
    with rev_class "override-unverified" and can never satisfy a strict
    pin match.
    """
    mid = entry["id"]
    if mid in overrides:
        path = Path(overrides[mid]).expanduser()
        if not (path / "config.json").is_file():
            return ("skipped", None, None,
                    f"--model-dir {path} has no config.json", None)
        return ("ready", path, None,
                "override: --model-dir content not verified against any "
                "revision; directory presence is not a pin",
                "override-unverified")

    repo_dir = model_repo_dir(hf_cache, entry["repo"])
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
            return ("skipped", None, None,
                    f"{entry['repo']} not present in local HF cache "
                    f"({sanitize(str(repo_dir))}); offline: not fetched", None)

    if entry.get("revision"):
        rev_dir = snaps / entry["revision"]
        if not (rev_dir / "config.json").is_file():
            have = [p.name for p in snaps.iterdir() if p.is_dir()]
            return ("skipped", None, None,
                    f"pinned revision {entry['revision']} not in cache; "
                    f"snapshots present: {have or 'none'}", None)
        return ("ready", rev_dir, entry["revision"],
                "pinned revision found in cache", "pinned")

    # revision null: resolve the factual snapshot from the cache itself.
    dirs = sorted(p for p in snaps.iterdir() if p.is_dir())
    if len(dirs) != 1:
        return ("skipped", None, None,
                f"revision unresolved and cache holds {len(dirs)} snapshots; "
                "refusing to guess", None)
    rev_dir = dirs[0]
    if not (rev_dir / "config.json").is_file():
        return ("skipped", None, None,
                f"snapshot {rev_dir.name} has no config.json", None)
    return ("ready", rev_dir, rev_dir.name,
            "resolved from local cache snapshot (factual)",
            "resolved-from-cache")


PROMPT_TOKENS_SNIPPET = """
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
p = tok.apply_chat_template([{"role": "user", "content": sys.argv[2]}],
                            add_generation_prompt=True, tokenize=True)
# transformers 5.x can return a BatchEncoding here; len() of it is the
# number of KEYS (2), not tokens - the exact bug that produced
# prompt_tokens=2 on every real M1 leg. Extract the id list itself.
if hasattr(p, "keys") and not isinstance(p, list):
    enc = p["input_ids"] if "input_ids" in p else getattr(p, "input_ids")
else:
    enc = p
if isinstance(enc, str):
    enc = tok(enc)["input_ids"]
ids = enc
while isinstance(ids, list) and ids and isinstance(ids[0], list):
    ids = ids[0]
print(json.dumps({"prompt_tokens": len(ids)}))
"""

PROVENANCE_SNIPPET = """
import json, sys
sys.path.insert(0, sys.argv[1])
from mlx_provenance import (installed_provenance, provenance_line,
                            harness_commit)
out = {
    "omarchy": installed_provenance(),
    "upstream_mlx": installed_provenance(dist_name="mlx"),
}
out["harness_commit"] = harness_commit()
out["line_omarchy"] = provenance_line(out["omarchy"])
print(json.dumps(out))
"""


def run_py(python, args, timeout=120):
    try:
        proc = subprocess.run(
            [python] + args, capture_output=True, text=True, timeout=timeout)
        return proc
    except (OSError, subprocess.SubprocessError) as exc:
        class _P:
            returncode, stdout, stderr = 127, "", f"{type(exc).__name__}: {exc}"
        return _P()


def probe_prompt_tokens(python, model_path, prompt_text):
    """Prompt token count with the model's own tokenizer, or None."""
    proc = run_py(python, ["-c", PROMPT_TOKENS_SNIPPET,
                           str(model_path), prompt_text])
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout.strip())["prompt_tokens"]
    except (ValueError, KeyError):
        return None


def probe_binary_provenance(python):
    proc = run_py(python, ["-c", PROVENANCE_SNIPPET, str(SCRIPTS_DIR)])
    if proc.returncode != 0:
        return {"error": (proc.stderr or "").strip()[-400:]}
    try:
        return json.loads(proc.stdout.strip())
    except ValueError:
        return {"error": "unparseable provenance output"}


def probe_packages(python):
    snippet = (
        "import importlib.metadata as m, json\n"
        "out = {}\n"
        "for d in ('mlx', 'mlx-omarchy', 'mlx-lm', 'transformers', 'numpy'):\n"
        "    try:\n"
        "        out[d] = m.version(d)\n"
        "    except Exception:\n"
        "        out[d] = None\n"
        "print(json.dumps(out))")
    proc = run_py(python, ["-c", snippet])
    if proc.returncode != 0:
        return {}
    try:
        return json.loads(proc.stdout.strip())
    except ValueError:
        return {}


# ------------------------------------------------------------- host facts

def _sysctl(key):
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _sw_vers(flag):
    try:
        out = subprocess.run(["sw_vers", flag], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def host_facts():
    sysname = platform.system()
    facts = {"system": sysname, "machine": platform.machine(),
             "python": platform.python_version()}
    if sysname == "Darwin":
        facts["chip"] = _sysctl("machdep.cpu.brand_string")
        facts["cores"] = _sysctl("hw.ncpu")
        mem = _sysctl("hw.memsize")
        facts["memsize_bytes"] = int(mem) if mem and mem.isdigit() else None
        vers = _sw_vers("-productVersion")
        build = _sw_vers("-buildVersion")
        facts["os"] = f"macOS {vers} ({build})" if vers else None
        try:
            sp = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                                capture_output=True, text=True, timeout=60)
            gpu = {}
            for line in sp.stdout.splitlines():
                s = line.strip()
                for key, field in (("Chipset Model", "chipset"),
                                   ("Total Number of Cores", "gpu_cores"),
                                   ("Metal Support", "metal")):
                    if s.startswith(key + ":") and field not in gpu:
                        gpu[field] = s.split(":", 1)[1].strip()
            facts["gpu"] = gpu or None
        except (OSError, subprocess.SubprocessError):
            facts["gpu"] = None
    else:
        facts["os"] = None
        rel = Path("/etc/os-release")
        if rel.is_file():
            for line in rel.read_text().splitlines():
                if line.startswith("PRETTY_NAME="):
                    facts["os"] = line.split("=", 1)[1].strip('"')
                    break
        facts["kernel"] = platform.release()
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            for line in cpuinfo.read_text().splitlines():
                if line.startswith("model name"):
                    facts["chip"] = line.split(":", 1)[1].strip()
                    break
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            m = re.search(r"MemTotal:\s+(\d+) kB", meminfo.read_text())
            if m:
                facts["memsize_bytes"] = int(m.group(1)) * 1024
        try:
            vi = subprocess.run(["vulkaninfo", "--summary"],
                                capture_output=True, text=True, timeout=30)
            drv = {}
            for line in vi.stdout.splitlines():
                s = line.strip()
                for key, field in (("driverName", "vulkan_driver"),
                                   ("deviceName", "device")):
                    if s.startswith(key) and field not in drv:
                        drv[field] = s.split("=", 1)[1].strip() if "=" in s else None
            facts["gpu"] = drv or None
        except (OSError, subprocess.SubprocessError):
            facts["gpu"] = None
    return facts


# pmset battery states with a known charging answer; anything else is unknown.
_BATTERY_STATES = frozenset({"charging", "discharging", "not charging", "charged"})


def power_state():
    """Power metadata, or None when unknown. Never inferred."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                                 text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        pct = re.search(r"(\d+)%", out)
        state = re.search(r"\d+%;\s*([^;]+);", out)
        label = state.group(1).strip() if state else None
        return {
            "raw": " ".join(out.split())[:200],
            "source": "AC Power" if "AC Power" in out else
                      ("Battery" if "Battery Power" in out else None),
            "percent": int(pct.group(1)) if pct else None,
            "charging": (label == "charging") if label in _BATTERY_STATES else None,
        }
    for ps in sorted(Path("/sys/class/power_supply").glob("*")):
        try:
            ptype = (ps / "type").read_text().strip()
            if ptype == "Mains":
                online = (ps / "online").read_text().strip()
                if online == "1":
                    return {"raw": f"{ps.name}: online", "source": "AC",
                            "percent": None, "charging": True}
            elif ptype == "Battery":
                cap = (ps / "capacity").read_text().strip()
                st = (ps / "status").read_text().strip()
                return {"raw": f"{ps.name}: {st} {cap}%", "source": "Battery",
                        "percent": int(cap) if cap.isdigit() else None,
                        "charging": st == "Charging"}
        except (OSError, ValueError):
            continue
    return None

def clean_check():
    """Model-serving processes that would contend for the accelerator.

    Status is contended | clean | unknown. Unknown only when the scan
    mechanism itself failed or saw no processes at all; an empty match
    list from a real scan is a clean result, not an unknown one.
    """
    found, scanned = [], 0
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["ps", "-axo", "pid=,comm="],
                                 capture_output=True, text=True,
                                 timeout=15).stdout
        except (OSError, subprocess.SubprocessError):
            return {"status": "unknown", "matched": [], "scanned": 0}
        for line in out.splitlines():
            parts = line.strip().split(None, 1)
            if not parts:
                continue
            scanned += 1
            if len(parts) == 2 and any(n in parts[1].lower()
                                       for n in CONTENDED_NAMES):
                found.append({"pid": parts[0], "comm": Path(parts[1]).name})
    else:
        try:
            proc_dirs = [p for p in Path("/proc").iterdir()
                         if p.name.isdigit()]
        except OSError:
            return {"status": "unknown", "matched": [], "scanned": 0}
        for proc_dir in proc_dirs:
            try:
                comm = (proc_dir / "comm").read_text().strip()
            except OSError:
                continue
            scanned += 1
            if any(n in comm.lower() for n in CONTENDED_NAMES):
                found.append({"pid": proc_dir.name, "comm": comm})
    status = "contended" if found else ("clean" if scanned else "unknown")
    return {"status": status, "matched": found, "scanned": scanned}


# ------------------------------------------------------------------ legs

def prompt_text(manifest, prompt_id):
    """Exact prompt string for a manifest prompt entry.

    "numbered" templates expand deterministically so macOS and Linux
    run byte-identical prompts without pasting kilobytes into the
    manifest.
    """
    entry = manifest["prompts"][prompt_id]
    if "text" in entry:
        return entry["text"]
    if entry.get("template") == "numbered":
        parts = [entry["base"]] + [
            f"{entry['item']} Entry {i} of {entry['items']}."
            for i in range(1, entry["items"] + 1)]
        return " ".join(parts)
    raise KeyError(f"prompt {prompt_id}: unsupported prompt entry")


def selected_workloads(manifest, selected):
    """Workloads in effect: all, or selection=explicit ones named."""
    wls = manifest["workloads"]
    if selected == "all":
        return wls, []
    chosen = set(selected)
    active, excluded = [], []
    for wl in wls:
        if wl.get("selection") == "explicit" and wl["id"] not in chosen:
            excluded.append(wl["id"])
        else:
            active.append(wl)
    return active, excluded
def pins_map(manifest, resolved):
    """Comparable pins per model: pinned and cache-resolved revisions.
    Override dirs carry no verified revision, so they appear only in
    models_resolved (revision null, class override-unverified) and are
    refused by check_pins, never listed as a pin."""
    by_id = {r["model_id"]: r for r in resolved}
    pins = {}
    for entry in manifest["models"]:
        r = by_id.get(entry["id"])
        if not r or r["status"] != "ready" or not r["revision"]:
            continue
        pins[entry["id"]] = {
            "revision": r["revision"],
            "class": r.get("revision_class") or
                     ("pinned" if entry.get("revision") else
                      "resolved-from-cache"),
        }
    return pins


def check_pins(resolved, expected):
    """Refuse strict pin match unless every expected model resolves to
    exactly the expected revision from trusted cache metadata. An
    --model-dir override has unverified provenance and is refused even
    when its directory name matches: directory names are not pins."""
    by_id = {r["model_id"]: r for r in resolved}
    problems = []
    for mid, rev in expected.items():
        r = by_id.get(mid)
        if not r or r["status"] != "ready":
            problems.append(f"{mid}: expected revision {rev} but model is "
                            "not ready on this machine")
        elif r.get("revision_class") == "override-unverified":
            problems.append(
                f"{mid}: --model-dir override has unverified provenance; "
                "it cannot satisfy a strict pin match")
        elif r["revision"] != rev:
            problems.append(f"{mid}: expected revision {rev}, this machine "
                            f"resolved {r['revision']} "
                            f"({r.get('revision_class')}); "
                            "same model id is not same weights")
    return "; ".join(problems) if problems else None


def build_legs(manifest, hf_cache, overrides, selected):
    workloads, excluded = selected_workloads(manifest, selected)
    resolved, legs, paths = [], [], {}
    for entry in manifest["models"]:
        status, path, rev, reason, rev_class = resolve_model(
            entry, hf_cache, overrides)
        resolved.append({"model_id": entry["id"], "repo": entry["repo"],
                         "revision": rev, "status": status,
                         "revision_class": rev_class,
                         "path": sanitize(str(path)) if path else None,
                         "detail": reason})
        if status != "ready":
            for wl in workloads:
                legs.append({"leg_id": f"{entry['id']}:{wl['id']}",
                             "model_id": entry["id"], "revision": rev,
                             "workload_id": wl["id"], "prompt_id": wl["prompt"],
                             "tokens": wl["tokens"], "status": "skipped",
                             "measured": False, "skip_reason": reason})
            continue
        paths[entry["id"]] = path
        for wl in workloads:
            legs.append({"leg_id": f"{entry['id']}:{wl['id']}",
                         "model_id": entry["id"], "revision": rev,
                         "workload_id": wl["id"], "prompt_id": wl["prompt"],
                         "tokens": wl["tokens"], "status": "ready",
                         "measured": False, "model_path": sanitize(str(path))})
    return resolved, legs, paths, excluded


def parse_bench_output(stdout):
    m = {"decode_tok_s": None, "decode_tokens": None, "requested_tokens": None,
         "prefill_s": None, "decode_mean_per_token_ms": None,
         "provenance_line": None, "generated_ids_sha256_16": None,
         "generated_ids_n": None, "prompt_tokens": None, "uclamp": None}
    result_json = None
    for line in stdout.splitlines():
        if result_json is None and line.startswith("{"):
            try:
                data = json.loads(line)
            except ValueError:
                data = None
            if isinstance(data, dict) and data.get("prefill_s"):
                result_json = data
        for regex, key, cast in ((DECODE_RE, None, float),
                                 (PREFILL_RE, "prefill_s", float),
                                 (PER_TOKEN_RE, "decode_mean_per_token_ms", float),
                                 (PROV_LINE_RE, "provenance_line", str),
                                 (UCLAMP_RE, "uclamp", str),
                                 (IDS_RE, None, str),
                                 (PROMPT_TOKENS_RE, None, str)):
            hit = regex.search(line)
            if not hit:
                continue
            if regex is DECODE_RE:
                m["decode_tok_s"] = float(hit.group(1))
                m["decode_tokens"] = int(hit.group(2))
                m["requested_tokens"] = int(hit.group(3))
            elif regex is IDS_RE:
                m["generated_ids_sha256_16"] = hit.group(1)
                m["generated_ids_n"] = int(hit.group(2))
            elif regex is PROMPT_TOKENS_RE:
                m["prompt_tokens"] = int(hit.group(1))
            else:
                m[key] = cast(hit.group(1))
    if result_json is not None:
        # bench_decode's JSON result line is the instrument for the
        # prefill span: prefill_s there is the monotonic-ns measurement
        # rounded to 6 decimals. The human "prefill" line is printed at
        # 3 decimals (1 ms), which on a 30-token prefill (~0.1 s) is up
        # to half a percent of pure rounding. Never re-derive timings
        # from printed output when the JSON beside it carries them.
        m["prefill_s"] = float(result_json["prefill_s"])
    return m


def execute_leg(leg, manifest, args, contended, paths):
    """Run one leg. `paths` holds RAW model dirs for execution; the leg
    carries only sanitized report data. A redacted path must never
    reach a subprocess: exec does not expand '~'."""
    prompt = prompt_text(manifest, leg["prompt_id"])
    gen = manifest["generation"]
    engine = SCRIPTS_DIR / Path(gen["engine_script"]).name
    cmd = [args.python, str(engine),
           "--model", str(paths[leg["model_id"]]),
           "--prompt", prompt,
           "--tokens", str(leg["tokens"]),
           "--temp", str(gen["temp"]),
           "--seed", str(gen["seed"]),
           "--warmup-tokens", str(gen.get("warmup_tokens", 4))]
    if args.wheel:
        cmd += ["--wheel", args.wheel]
    env = dict(os.environ)
    env.update(gen.get("env", {}))
    if args.allow_non_apple:
        env["MLX_OMARCHY_ALLOW_NON_APPLE"] = "1"
    leg["contended"] = contended
    leg["cmd"] = sanitize(" ".join(cmd))
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.timeout, env=env)
    except subprocess.TimeoutExpired:
        leg.update(status="failed", exit_code=None,
                   stderr_tail=f"timed out after {args.timeout}s",
                   duration_s=round(time.monotonic() - start, 2))
        return leg
    except OSError as exc:
        leg.update(status="failed", exit_code=127,
                   stderr_tail=f"{type(exc).__name__}: {exc}",
                   duration_s=round(time.monotonic() - start, 2))
        return leg
    leg["duration_s"] = round(time.monotonic() - start, 2)
    leg["exit_code"] = proc.returncode
    if proc.returncode != 0:
        leg.update(status="failed",
                   stderr_tail=(proc.stderr or "").strip()[-500:])
        return leg
    metrics = parse_bench_output(proc.stdout)
    # Leg agreement: the output must be about THIS leg. bench_decode
    # prints the rate over produced-1 tokens (the first timed token
    # carries the prefill), so a 4-token leg must report
    # requested 4, decode span 3. Anything else is not this leg's
    # number and is refused, never reported as measured.
    n = leg["tokens"]
    bad = []
    if metrics["decode_tok_s"] is None:
        bad.append("no decode rate in output")
    else:
        if not (math.isfinite(metrics["decode_tok_s"])
                and metrics["decode_tok_s"] > 0):
            bad.append(f"decode rate not finite-positive: "
                       f"{metrics['decode_tok_s']}")
        if metrics["requested_tokens"] != n:
            bad.append(f"requested_tokens {metrics['requested_tokens']} "
                       f"!= leg tokens {n}")
        if metrics["decode_tokens"] != n - 1:
            bad.append(f"decode_tokens {metrics['decode_tokens']} != "
                       f"tokens-1 ({n - 1})")
    if (metrics["prefill_s"] is None
            or not math.isfinite(metrics["prefill_s"])
            or metrics["prefill_s"] <= 0):
        bad.append("prefill seconds missing or not finite-positive")
    if not metrics["provenance_line"]:
        bad.append("provenance line missing")
    # generated_ids digest is mandatory on every measured leg: the
    # integration bench_decode prints it (generated_ids sha256:<16hex>
    # n=N), and cross-platform identity is compared on token ids.
    digest = metrics["generated_ids_sha256_16"]
    if not digest or len(digest) != 16 or \
            any(c not in "0123456789abcdef" for c in digest):
        bad.append("generated_ids digest missing or not 16 hex")
    elif metrics["generated_ids_n"] != n:
        bad.append(f"generated_ids n {metrics['generated_ids_n']} != "
                   f"tokens {n}")
    if bad:
        leg.update(status="failed",
                   stderr_tail="leg-agreement validation failed: "
                   + "; ".join(bad))
        return leg
    # prompt_tokens is trusted only from bench_decode's own measured
    # GenerationResponse count. The independent tokenizer probe is gone:
    # transformers 5.x made len(apply_chat_template(...)) return the
    # BatchEncoding key count (2), which poisoned every derived prefill
    # tok/s on the real M1 run. Without the bench-reported count the
    # derived rate stays null - a guess is never substituted.
    if metrics["prompt_tokens"]:
        metrics["prefill_tok_s"] = round(
            metrics["prompt_tokens"] / metrics["prefill_s"], 3)
        metrics["prefill_tok_s_basis"] = (
            "bench_decode prompt_tokens / prefill seconds")
    leg["prompt_tokens"] = metrics["prompt_tokens"]
    leg.update(status="measured", measured=True, metrics=metrics)
    return leg


# -------------------------------------------------------------- assembly

def collect_metadata(args, manifest):
    home = os.path.expanduser("~")
    prov = probe_binary_provenance(args.python)
    for section in ("omarchy", "upstream_mlx"):
        block = prov.get(section)
        if isinstance(block, dict) and block.get("files"):
            block["files"] = [
                {**f, "path": sanitize(f["path"], home)} for f in block["files"]]
    return {
        "host": host_facts(),
        "packages": probe_packages(args.python),
        "binary_provenance": prov,
        "power": power_state(),
        "clean_check": clean_check(),
        "source": {"harness_commit": prov.get("harness_commit"),
                   "note": "commit captured by scripts/mlx_provenance.py"},
    }


def main():
    import caps_sim_guard
    caps_sim_guard.refuse_if_simulated("bench_matrix")
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(SCRIPTS_DIR / "bench_matrix.json"))
    ap.add_argument("--mode", choices=("plan", "metadata", "run"),
                    default="plan")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter that runs the benchmark and probes "
                         "(the mlx / mlx-lm venv python)")
    ap.add_argument("--hf-cache", default=os.environ.get(
        "HF_HUB_CACHE",
        os.path.join(os.path.expanduser("~"),
                     ".cache", "huggingface", "hub")))
    ap.add_argument("--model-dir", action="append", default=[],
                    metavar="ID=PATH",
                    help="use this local model dir for manifest model ID")
    ap.add_argument("--wheel", default=None,
                    help="passed to bench_decode --wheel for its provenance "
                         "gate")
    ap.add_argument("--allow-non-apple", action="store_true",
                    help="set MLX_OMARCHY_ALLOW_NON_APPLE=1 for the child "
                         "(development/software Vulkan only; recorded)")
    ap.add_argument("--host-label", default=None,
                    help="free-text host label; the hostname is never "
                         "auto-detected")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per-leg timeout in seconds")
    ap.add_argument("--select", action="append", default=[],
                    metavar="WORKLOAD_ID",
                    help="include an explicit-selection workload; 'all' "
                         "selects everything")
    ap.add_argument("--expect-pins", action="append", default=[],
                    metavar="MODEL_ID=REVISION",
                    help="refuse to run (exit 4) unless each model resolves "
                         "to exactly this revision; pass the other "
                         "machine's pins map for cross-machine comparison")
    ap.add_argument("--out", default=None, help="also write JSON here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    overrides = dict(
        part.split("=", 1) for part in args.model_dir if "=" in part)
    try:
        manifest = json.loads(Path(args.manifest).read_text())
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read manifest: {exc}", file=sys.stderr)
        sys.exit(1)
    if manifest.get("schema") != SCHEMA:
        print(f"ERROR: manifest schema {manifest.get('schema')!r} != {SCHEMA!r}",
              file=sys.stderr)
        sys.exit(1)

    result = {"schema": SCHEMA, "mode": args.mode,
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
              "host_label": args.host_label,
              "manifest": sanitize(args.manifest)}
    if args.mode == "metadata":
        result.update(collect_metadata(args, manifest))
        _emit(result, args.out)
        return

    selected = "all" if "all" in args.select else args.select
    resolved, legs, paths, excluded = build_legs(
        manifest, args.hf_cache, overrides, selected)
    result["models_resolved"] = resolved
    result["pins"] = pins_map(manifest, resolved)
    if excluded:
        result["notes"] = [
            f"explicit-selection workloads not selected, no legs built: "
            f"{', '.join(excluded)} (use --select {excluded[0]} or "
            "--select all)"]

    if args.mode == "plan":
        for leg in legs:
            if leg["status"] != "ready":
                continue
            count = probe_prompt_tokens(
                args.python, paths[leg["model_id"]],
                prompt_text(manifest, leg["prompt_id"]))
            leg["prompt_tokens"] = count
            leg["prompt_tokens_probe"] = (
                "transformers tokenizer" if count is not None
                else "unavailable; recorded null, never assumed")
        result["legs"] = legs
        _emit(result, args.out)
        return

    # run mode
    expected = dict(
        part.split("=", 1) for part in args.expect_pins if "=" in part)
    if expected:
        refusal = check_pins(result["models_resolved"], expected)
        if refusal:
            result["refusal"] = refusal
            _emit(result, args.out)
            sys.exit(4)
    meta = collect_metadata(args, manifest)
    result.update({k: meta[k] for k in
                   ("host", "packages", "binary_provenance", "power",
                    "clean_check", "source")})
    contended = meta["clean_check"]["status"] == "contended"
    if contended:
        result["warning"] = (
            "contended: " + ", ".join(
                f"{p['comm']}(pid {p['pid']})"
                for p in meta["clean_check"]["matched"]) +
            " running; timings are contended, not clean")
    if args.allow_non_apple:
        result["env"] = {"MLX_OMARCHY_ALLOW_NON_APPLE": "1 (dev/software GPU)"}
    env_ok = run_py(args.python, ["-c", "import mlx_lm"]).returncode == 0
    measured = 0
    for leg in legs:
        if leg["status"] != "ready":
            continue
        if not env_ok:
            leg.update(status="skipped",
                       skip_reason=("mlx_lm not importable in "
                                    f"{sanitize(args.python)}; environment "
                                    "incomplete, nothing executed"))
            continue
        execute_leg(leg, manifest, args, contended, paths)
        if leg["status"] == "measured":
            measured += 1
    result["legs"] = legs
    result["measured_legs"] = measured
    _emit(result, args.out)
    if measured == 0:
        sys.exit(3)


def _emit(result, out_path):
    text = json.dumps(result, indent=2)
    print(text)
    if out_path:
        Path(out_path).write_text(text + "\n")


def self_test():
    """No-GPU checks: parsing, skip classification, sanitization."""
    out = "decode 1.97 tok/s over 64 tokens (64 requested, EOS suppressed)\n" \
          "prefill 0.512s (reported separately, excluded from decode)\n" \
          "decode mean per-token 507.6 ms\n" \
          "provenance: mlx-omarchy 0.32.2 mx=0.32.2 verified=match " \
          "harness=abc1234 x=1\n" \
          "generated_ids sha256:0123456789abcdef n=64 first=1,2 last=3,4\n"
    m = parse_bench_output(out)
    assert m["decode_tok_s"] == 1.97 and m["decode_tokens"] == 64
    assert m["requested_tokens"] == 64 and m["prefill_s"] == 0.512
    assert m["decode_mean_per_token_ms"] == 507.6
    assert m["provenance_line"].startswith("mlx-omarchy")
    assert m["generated_ids_sha256_16"] == "0123456789abcdef"
    assert m["generated_ids_n"] == 64
    assert m["uclamp"] is None  # no uclamp line: an older bench_decode.py

    # The clamp outcome is persisted: applied, switched off, or refused.
    for line, want in (("uclamp: uclamp_min=1024", "uclamp_min=1024"),
                       ("uclamp: MLX_OMARCHY_UCLAMP_MIN off or invalid",
                        "MLX_OMARCHY_UCLAMP_MIN off or invalid"),
                       ("uclamp: perf_placement not found, no clamp",
                        "perf_placement not found, no clamp")):
        assert parse_bench_output(line + "\n" + out)["uclamp"] == want, line

    # The JSON result line overrides the 3-decimal printed span: the
    # printed "prefill 0.512s" would quantize a 0.512345 s span.
    out_json = out + '{"decode_tps": 1.97, "generated": 64, ' \
        '"ids_sha256_16": "0123456789abcdef", ' \
        '"prefill_s": 0.512345, "prompt_tokens": 30, ' \
        '"prefill_tps": 58.5549, "device": "gpu", ' \
        '"engine": "bench_decode", "ids_first": [1, 2], ' \
        '"ids_last": [3, 4]}\n'
    mj = parse_bench_output(out_json)
    assert mj["prefill_s"] == 0.512345, mj["prefill_s"]
    assert mj["decode_tok_s"] == 1.97 and mj["decode_tokens"] == 64

    # 4-token fixture: the shape a tokens=4 leg actually produces.
    out4 = "decode 7.88 tok/s over 3 tokens (4 requested, EOS suppressed)\n" \
           "prefill 0.512s (reported separately, excluded from decode)\n" \
           "decode mean per-token 128.2 ms\n" \
           "provenance: mlx-omarchy 0.32.2 mx=0.32.2 verified=match " \
           "harness=abc1234 x=1\n" \
           "generated_ids sha256:0123456789abcdef n=4 first=1,2 last=3,4\n"

    class FakeOut:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    # cc3897f false-pass: a 64-token output on a tokens=4 leg must be
    # refused by leg-agreement validation, never reported as measured.
    mismatch = {"leg_id": "x", "model_id": "m", "model_path": "/m",
                "prompt_id": "s", "tokens": 4, "measured": False}
    manifest = {"prompts": {"s": {"text": "Hi"}},
                "generation": {"temp": 0, "seed": 0, "warmup_tokens": 4,
                               "engine_script": "bench_decode.py"}}
    real_run = subprocess.run
    subprocess.run = lambda *a, **k: FakeOut(out)
    try:
        class A:
            python, wheel, allow_non_apple, timeout = "py", None, False, 10
        execute_leg(mismatch, manifest, A(), contended=False, paths={"m": Path("/m")})
    finally:
        subprocess.run = real_run
    assert mismatch["status"] == "failed" and not mismatch["measured"]
    assert "leg-agreement" in mismatch["stderr_tail"], mismatch["stderr_tail"]
    assert mismatch["exit_code"] == 0  # exited fine; numbers were not ours

    # Digest is mandatory: integration bench_decode always prints it
    # (1a96615). An output without it is a failed leg, never measured.
    out4_nodigest = out4.replace(
        "generated_ids sha256:0123456789abcdef n=4 first=1,2 last=3,4\n",
        "")
    nodigest = {"leg_id": "x", "model_id": "m", "model_path": "/m",
                "prompt_id": "s", "tokens": 4, "measured": False}
    subprocess.run = lambda *a, **k: FakeOut(out4_nodigest)
    try:
        class A2:
            python, wheel, allow_non_apple, timeout = "py", None, False, 10
        execute_leg(nodigest, manifest, A2(), contended=False,
                    paths={"m": Path("/m")})
    finally:
        subprocess.run = real_run
    assert nodigest["status"] == "failed" and not nodigest["measured"]
    assert "digest" in nodigest["stderr_tail"], nodigest["stderr_tail"]

    # prompt_tokens trusted only from bench output; the derived prefill
    # rate exists only when the bench reported the count (M1 prompt_
    # tokens=2 probe bug).
    out4pt = out4 + "prompt_tokens 30\n"
    withpt = {"leg_id": "x", "model_id": "m", "model_path": "/m",
              "prompt_id": "s", "tokens": 4, "measured": False}
    subprocess.run = lambda *a, **k: FakeOut(out4pt)
    try:
        execute_leg(withpt, manifest, A2(), contended=False,
                    paths={"m": Path("/m")})
    finally:
        subprocess.run = real_run
    assert withpt["status"] == "measured"
    assert withpt["prompt_tokens"] == 30
    assert withpt["metrics"]["prefill_tok_s"] == round(30 / 0.512, 3)
    assert withpt["metrics"]["prefill_tok_s_basis"].startswith(
        "bench_decode prompt_tokens")
    nopt = {"leg_id": "x2", "model_id": "m", "model_path": "/m",
            "prompt_id": "s", "tokens": 4, "measured": False}
    subprocess.run = lambda *a, **k: FakeOut(out4)
    try:
        execute_leg(nopt, manifest, A2(), contended=False,
                    paths={"m": Path("/m")})
    finally:
        subprocess.run = real_run
    assert nopt["status"] == "measured"
    assert nopt["prompt_tokens"] is None
    assert "prefill_tok_s" not in nopt["metrics"]

    assert sanitize(f"{os.path.expanduser('~')}/models/q4") == "~/models/q4"
    assert "hostname" not in json.dumps(
        {"host": host_facts()} | {"power": power_state()})

    import io
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        entry = {"id": "m", "repo": "org/none", "revision": None}
        status, _, rev, reason, _ = resolve_model(entry, tmp, {})
        assert status == "skipped" and rev is None and \
            "not present" in reason

        snaps = Path(tmp) / "models--org--x" / "snapshots"
        (snaps / "aaaa").mkdir(parents=True)
        entry2 = {"id": "m2", "repo": "org/x", "revision": "deadbeef"}
        status2, _, _, reason2, cls2 = resolve_model(entry2, tmp, {})
        assert status2 == "skipped" and "pinned revision" in reason2
        assert cls2 is None

        (snaps / "deadbeef" / "config.json").parent.mkdir(parents=True)
        (snaps / "deadbeef" / "config.json").write_text("{}")
        status3, path3, rev3, _, cls3 = resolve_model(entry2, tmp, {})
        assert status3 == "ready" and rev3 == "deadbeef"
        assert path3.name == "deadbeef" and cls3 == "pinned"

        entry4 = {"id": "m4", "repo": "org/x", "revision": None}
        status4, _, rev4, reason4, _ = resolve_model(entry4, tmp, {})
        assert status4 == "skipped" and "refusing to guess" in reason4

        # MatrixReview false-pass: an arbitrary dir with a config.json
        # must never be accepted as the known SHA. Override reports
        # revision null + override-unverified and check_pins refuses it.
        bogus = Path(tmp) / "bogus-weights"
        bogus.mkdir()
        (bogus / "config.json").write_text("{}")
        entry5 = {"id": "m5", "repo": "org/x", "revision": "deadbeef"}
        status5, path5, rev5, reason5, cls5 = resolve_model(
            entry5, tmp, {"m5": str(bogus)})
        assert status5 == "ready" and rev5 is None, (rev5, reason5)
        assert cls5 == "override-unverified" and "not a pin" in reason5
        resolved5 = [{"model_id": "m5", "status": "ready",
                      "revision": None, "revision_class": cls5}]
        refusal5 = check_pins(resolved5, {"m5": "deadbeef"})
        assert refusal5 and "strict pin match" in refusal5

    mfix = {"prompts": {"n": {"template": "numbered", "base": "B",
                              "item": "C", "items": 2},
                        "t": {"text": "T"}},
            "workloads": [
                {"id": "w1", "prompt": "t", "tokens": 4},
                {"id": "wx", "prompt": "n", "tokens": 4,
                 "selection": "explicit"}]}
    assert prompt_text(mfix, "n") == "B C Entry 1 of 2. C Entry 2 of 2."
    assert prompt_text(mfix, "t") == "T"
    active, excluded = selected_workloads(mfix, [])
    assert [w["id"] for w in active] == ["w1"] and excluded == ["wx"]
    active, excluded = selected_workloads(mfix, ["wx"])
    assert [w["id"] for w in active] == ["w1", "wx"] and excluded == []
    active, excluded = selected_workloads(mfix, "all")
    assert excluded == []

    resolved = [{"model_id": "m1", "status": "ready", "revision": "aaa",
                 "revision_class": "pinned"},
                {"model_id": "m2", "status": "ready", "revision": "bbb",
                 "revision_class": "resolved-from-cache"}]
    man = {"models": [{"id": "m1", "revision": "aaa"},
                      {"id": "m2", "revision": None}]}
    pins = pins_map(man, resolved)
    assert pins == {"m1": {"revision": "aaa", "class": "pinned"},
                    "m2": {"revision": "bbb",
                           "class": "resolved-from-cache"}}
    assert check_pins(resolved, {"m2": "bbb"}) is None
    refusal = check_pins(resolved, {"m2": "ccc"})
    assert refusal and "not same weights" in refusal
    refusal2 = check_pins(resolved, {"gone": "ddd"})
    assert refusal2 and "not ready on this machine" in refusal2

    with tempfile.TemporaryDirectory() as tmp_home:
        model_dir = Path(tmp_home) / "cache" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        manifest = {"prompts": {"s": {"text": "Hi"}},
                    "generation": {"temp": 0, "seed": 0, "warmup_tokens": 4,
                                   "engine_script": "bench_decode.py"}}
        leg = {"leg_id": "t", "model_id": "m", "prompt_id": "s",
               "tokens": 4, "model_path": "~/cache/model"}
        class FakeProc:
            returncode = 0
            stdout = out4
            stderr = ""
        received = {}
        real_run = subprocess.run


        def spy_run(cmd, **kwargs):
            if "--model" in cmd:
                received["cmd"] = list(cmd)
            return FakeProc()
        subprocess.run = spy_run
        real_expanduser = os.path.expanduser
        os.path.expanduser = lambda _: tmp_home
        try:
            class A:
                python, wheel, allow_non_apple, timeout = \
                    "py", None, False, 10
            execute_leg(leg, manifest, A(), contended=True,
                        paths={"m": model_dir})
        finally:
            subprocess.run = real_run
            os.path.expanduser = real_expanduser

        # The process must receive the REAL path, and it must exist on
        # disk; a literal '~' path is exactly the M1 false-failure bug.
        model_arg = received["cmd"][received["cmd"].index("--model") + 1]
        assert not model_arg.startswith("~"), model_arg
        assert Path(model_arg).is_dir() and \
            Path(model_arg, "config.json").is_file()
        # The report carries only the redacted form.
        assert leg["model_path"] == "~/cache/model", leg["model_path"]
        assert leg["status"] == "measured" and leg["contended"] is True
        assert leg["metrics"]["decode_tok_s"] == 7.88
        assert leg["metrics"]["decode_tokens"] == 3
        assert leg["metrics"]["requested_tokens"] == 4
        assert leg.get("prompt_tokens") is None  # spy stdout is not JSON

    # Regression (16m1 metadata run): sw_vers must be invoked as its own
    # command, never as a sysctl oid, or host.os parses null on macOS.
    real_system = platform.system

    def fake_run(cmd, **kwargs):
        class P:
            returncode = 0
            stderr = ""
        if cmd[:2] == ["sw_vers", "-productVersion"]:
            P.stdout = "26.6.2\n"
        elif cmd[:2] == ["sw_vers", "-buildVersion"]:
            P.stdout = "25G83\n"
        elif cmd[:2] == ["sysctl", "-n"] and "brand_string" in cmd[2]:
            P.stdout = "Apple M1 Max\n"
        else:
            P.stdout, P.returncode = "", 1
        return P()

    subprocess.run = fake_run
    platform.system = lambda: "Darwin"
    try:
        mac = host_facts()
    finally:
        subprocess.run = real_run
        platform.system = real_system
    assert mac["os"] == "macOS 26.6.2 (25G83)", mac["os"]
    assert mac["chip"] == "Apple M1 Max"

    # Main() run path, end to end with faked subprocesses: a measured
    # leg must increment measured_legs. Guards the lost 'measured = 0'
    # initialization (MatrixReview P0) that crashed every real run.
    with tempfile.TemporaryDirectory() as tmp_home:
        model_dir = Path(tmp_home) / "cache" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        man = {"schema": SCHEMA,
               "generation": {"temp": 0, "seed": 0, "warmup_tokens": 4,
                              "engine": "bench_decode",
                              "engine_script": "bench_decode.py",
                              "env": {}},
               "models": [{"id": "m", "repo": "org/x", "revision": None,
                           "optional": True}],
               "prompts": {"s": {"text": "Hi"}},
               "workloads": [{"id": "w", "prompt": "s", "tokens": 4}]}
        man_path = Path(tmp_home) / "manifest.json"
        out_path = Path(tmp_home) / "out.json"
        man_path.write_text(json.dumps(man))
        real_argv = sys.argv
        sys.argv = ["bench_matrix.py", "--mode", "run",
                    "--manifest", str(man_path),
                    "--model-dir", f"m={model_dir}",
                    "--python", "py", "--host-label", "selftest",
                    "--out", str(out_path)]
        def main_spy(cmd, **kwargs):
            class P:
                returncode = 0
                stderr = ""
            P.stdout = out4 if isinstance(cmd, list) and \
                "--model" in cmd else ""
            return P()

        real_stdout = sys.stdout
        sys.stdout = io.StringIO()
        subprocess.run = main_spy
        try:
            main()
        finally:
            subprocess.run = real_run
            sys.stdout = real_stdout
            sys.argv = real_argv
        emitted = json.loads(out_path.read_text())
        assert emitted["measured_legs"] == 1, emitted["measured_legs"]
        assert emitted["legs"][0]["status"] == "measured"

    print("self-test: OK (parsing, skip rules, templates, selection, pin "
          "refusal, sanitization, leg execution, darwin os parse, "
          "main run path)")


if __name__ == "__main__":
    main()
