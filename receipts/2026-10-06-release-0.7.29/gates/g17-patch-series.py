#!/usr/bin/env python3
"""g17-patch-series: run apply-mlx-lm-patches.sh against PRISTINE mlx-lm
0.31.3 and 0.32.0 trees built like tests/test_apply_mlxlm_series.py:
pip download --no-deps + unzip (no index/install needed on the host).
APPLY_SCRIPT is REQUIRED (no defaults). PASS = both versions: apply rc=0
AND an idempotent second run rc=0. CPU-only."""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile

APPLY = os.environ.get("APPLY_SCRIPT")
if not APPLY or not os.path.isfile(APPLY):
    sys.exit("APPLY_SCRIPT is required (path to apply-mlx-lm-patches.sh)")

LINES = ("0.31.3", "0.32.0")
results = {}
failed = False
work = tempfile.mkdtemp(prefix="g17-")
for ver in LINES:
    case = {"version": ver}
    td = os.path.join(work, ver.replace(".", ""))
    os.makedirs(td, exist_ok=True)
    r = subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps",
                        "-q", "-d", td, f"mlx-lm=={ver}"],
                       capture_output=True, text=True)
    case["download_rc"] = r.returncode
    if r.returncode != 0:
        case["error"] = r.stderr[-300:]
        results[ver] = case
        failed = True
        continue
    wheels = [f for f in os.listdir(td) if f.endswith(".whl")]
    if not wheels:
        case["error"] = "no wheel downloaded"
        results[ver] = case
        failed = True
        continue
    tree = os.path.join(td, "lib", "python3.11", "site-packages")
    os.makedirs(tree, exist_ok=True)
    with zipfile.ZipFile(os.path.join(td, wheels[0])) as zf:
        zf.extractall(tree)
    env = dict(os.environ, MLX_OMARCHY_CONV_RING="0")
    # APPLY expects the VENV ROOT (it appends lib/python3.*/site-packages)
    a1 = subprocess.run(["bash", APPLY, td], env=env,
                        capture_output=True, text=True)
    a2 = subprocess.run(["bash", APPLY, td], env=env,
                        capture_output=True, text=True)
    case["apply1_rc"] = a1.returncode
    case["apply2_rc"] = a2.returncode
    ok = a1.returncode == 0 and a2.returncode == 0
    case["pass"] = ok
    if not ok:
        case["stderr_tail"] = (a1.stderr or a2.stderr)[-300:]
        failed = True
    results[ver] = case
    print(ver, "apply1", a1.returncode, "apply2", a2.returncode,
          "PASS" if ok else "FAIL", file=sys.stderr)

summary = {"gate": "g17-patch-series", "apply_script": APPLY,
           "results": results, "pass": not failed}
print(json.dumps(summary, indent=2))
pathlib.Path("g17-result.json").write_text(json.dumps(summary, indent=2))
sys.exit(0 if not failed else 1)
