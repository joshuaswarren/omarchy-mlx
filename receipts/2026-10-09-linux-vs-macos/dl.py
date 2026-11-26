"""dl.py <key> <kind> <repo> [file]: download one model into the HF cache, write dl/<key>.path then dl/<key>.ok (or dl/<key>.fail)."""
import os
import sys
import time

key, kind, repo = sys.argv[1:4]
fn = sys.argv[4] if len(sys.argv) > 4 else ""
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dl")
os.makedirs(d, exist_ok=True)
t0 = time.time()
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    path = hf_hub_download(repo, fn) if kind == "gguf" else snapshot_download(repo)
    with open(os.path.join(d, key + ".path"), "w") as f:
        f.write(path)
    with open(os.path.join(d, key + ".ok"), "w") as f:
        f.write("%d s\n" % (time.time() - t0))
    print("DL ok", key, "%d s" % (time.time() - t0), path)
except Exception as e:  # recorded, the runner reports the row as not run
    with open(os.path.join(d, key + ".fail"), "w") as f:
        f.write(repr(e)[:300])
    print("DL FAIL", key, repr(e)[:200])
