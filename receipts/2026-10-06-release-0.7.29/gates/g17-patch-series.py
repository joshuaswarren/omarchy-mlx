#!/usr/bin/env python3
"""g17-patch-series: run apply-mlx-lm-patches.sh against PRISTINE mlx-lm
0.31.3 and 0.32.0 venvs. PASS = both rc=0 AND a second (idempotent) run
rc=0 on each. CPU-only."""
import subprocess, sys, tempfile, os, pathlib, time

APPLY = os.environ.get(
    "APPLY_SCRIPT",
    os.path.expanduser(
        "~/v0.7.30-repo/scripts/apply-mlx-lm-patches.sh"))
LINES = {"0.31.3": "0.31.3", "0.32.0": "0.32.0"}
results = {}
failed = False
for ver in ("0.31.3", "0.32.0"):
    venv = os.path.join(tempfile.mkdtemp(prefix=f"g17-{ver}-"), "venv")
    subprocess.run([sys.executable, "-m", "venv", "--clear", venv], check=True)
    r = subprocess.run([f"{venv}/bin/pip", "install", "-q", "--no-index",
                        f"mlx-lm=={ver}"], capture_output=True, text=True)
    if r.returncode != 0:
        results[ver] = {"install": "FAIL", "apply1": "n/a", "apply2": "n/a"}
        failed = True
        continue
    env = dict(os.environ, MLX_OMARCHY_CONV_RING="0")
    a1 = subprocess.run(["bash", APPLY, venv], env=env, capture_output=True, text=True)
    a2 = subprocess.run(["bash", APPLY, venv], env=env, capture_output=True, text=True)
    results[ver] = {"apply1_rc": a1.returncode, "apply2_rc": a2.returncode,
                    "apply2_tail": a2.stdout.splitlines()[-1] if a2.stdout else ""}
    ok = a1.returncode == 0 and a2.returncode == 0
    results[ver]["pass"] = ok
    failed |= not ok
    print(ver, "apply1", a1.returncode, "apply2", a2.returncode, "PASS" if ok else "FAIL")
import json
print(json.dumps({"gate": "g17-patch-series", "results": results,
                  "pass": not failed}, indent=2))
sys.exit(0 if not failed else 1)
