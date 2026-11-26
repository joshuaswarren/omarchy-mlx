"""Provenance gate, then the repository's scripts/profile_generate.py.

usage: trace_gen.py <worktree> <profile_generate args...>
Run with MLX_OMARCHY_GPU_PROFILE=<path> on a profiling (diag) wheel.
"""
import runpy
import sys

import prov

prov.check()
wt = sys.argv[1]
sys.argv = [f"{wt}/scripts/profile_generate.py"] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
