#!/usr/bin/env python3
"""omarchy_syntax_check: compile-gate driver for CI.

Configures nothing and links nothing: it reads compile_commands.json from
a configured build dir, selects the omarchy backend translation units,
rewrites each compile to -fsyntax-only, and runs them in parallel. This
is the deep counterpart of scripts/check_compute_kernel_wiring.py: the
text gate catches enum/dispatch/shader drift in about a second; this
catches anything only a real C++ frontend sees (missing declarations,
type errors, bad includes) in a couple of minutes on a 4-core runner.

Limitation: it validates the preprocessor state of the configured build.
With glslang (no GL_EXT_bfloat16) the runner configures with
MLX_OMARCHY_BF16_DIRECT off, so the bf16 direct GEMM compile path is
validated only by real shaderc/glslc builds (dev boxes, jw16 gates).

Usage:
  omarchy_syntax_check.py <build_dir> [--jobs N]
Exit: 0 all translation units pass, 1 any failure / no units found.
"""
import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OMARCHY_MARKER = "/mlx/backend/omarchy/"


def syntax_only_command(entry):
    """Rewrite one compile_commands entry into a -fsyntax-only command."""
    cmd = entry["command"]
    cmd = re.sub(r" -o \S+", "", cmd)
    cmd = re.sub(r" -M[DTF] \S+", "", cmd)  # depfile emission flags
    return cmd + " -fsyntax-only", entry["directory"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    db_path = Path(args.build_dir) / "compile_commands.json"
    if not db_path.is_file():
        sys.exit("omarchy-syntax-check: %s not found (configure first)" % db_path)
    entries = [
        e for e in json.loads(db_path.read_text())
        if OMARCHY_MARKER in e["file"]
    ]
    if not entries:
        sys.exit("omarchy-syntax-check: no omarchy translation units in %s"
                 % db_path)

    jobs = [(e["file"],) + syntax_only_command(e) for e in entries]

    def run(job):
        src, cmd, cwd = job
        p = subprocess.run(cmd, shell=True, cwd=cwd,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return src, p.returncode, p.stderr.decode("utf-8", "replace")

    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for src, rc, err in pool.map(run, jobs):
            name = src.split(OMARCHY_MARKER)[-1]
            if rc == 0:
                print("ok   %s" % name)
            else:
                failures.append(name)
                print("FAIL %s\n%s" % (name, err))
    print("omarchy-syntax-check: %d/%d translation units pass"
          % (len(jobs) - len(failures), len(jobs)))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
