"""CPU-disabled wheel control: a GPU add runs, an explicit CPU-stream add is refused.
usage: cpu_refused.py EXPECT_WHEEL REPO_ROOT"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[2]) / "scripts"))
import mlx_provenance as mp
prov = mp.installed_provenance(expect_wheel=Path(sys.argv[1])); print(mp.provenance_line(prov), flush=True)
if prov["verified"] != "match" or prov["version_match"] is not True:
    sys.exit(3)
import mlx.core as mx
a = mx.ones((4,)); mx.eval(a + a); print("gpu add ok", (a + a).tolist())
try:
    mx.eval(mx.add(a, a, stream=mx.cpu)); print("CPU ADD RAN"); sys.exit(1)
except Exception as e:
    print("cpu add refused:", type(e).__name__, str(e)[:160])
