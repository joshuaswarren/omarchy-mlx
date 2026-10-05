import pathlib, shutil, hashlib, sys

# FamMinimaxM3 scratch cleanup on the M2 (allowlist: exactly these two dirs).
targets = [pathlib.Path("/var/tmp/fmm3-venv"), pathlib.Path("/var/tmp/fmm3-wheel")]
for t in targets:
    if t.exists():
        print("deleting", t)
        shutil.rmtree(t)
    else:
        print("already gone:", t)
print("remaining fmm3 paths:", list(pathlib.Path("/var/tmp").glob("fmm3*")))
