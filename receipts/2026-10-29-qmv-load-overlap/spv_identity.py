"""Gate 5: SPIR-V module identity between the v0.7.34 tag tree and the load-overlap branch, same glslc binary on both sides.

Compiles every omarchy_shader() target whose source is qmm_vec.comp (the only file the branch changes) with exactly the flags CMake uses
(glslc -O --target-env=vulkan1.3 <defines>) from both trees, prints the glslc version, and one line per module: SAME or DIFF with the sha256 prefixes.
Also confirms that the two trees differ in no other file under overlay/mlx/backend/omarchy/shaders. Usage: spv_identity.py REPO TAG_SHA BRANCH_SHA WORKDIR
"""
import hashlib
import os
import re
import subprocess
import sys

repo, tag_sha, branch_sha, work = sys.argv[1:5]


def run(*cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


print("GLSLC", run("glslc", "--version").splitlines()[0:2], flush=True)
changed = run("git", "-C", repo, "diff", "--name-only", tag_sha, branch_sha, "--", "overlay/mlx/backend/omarchy").split()
print("CHANGED-FILES", changed, flush=True)
trees = {}
for name, sha in (("tag", tag_sha), ("branch", branch_sha)):
    d = os.path.join(work, name)
    os.makedirs(d, exist_ok=True)
    archive = subprocess.run(["git", "-C", repo, "archive", sha, "overlay/mlx/backend/omarchy"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-C", d, "-xf", "-"], input=archive, check=True)
    trees[name] = os.path.join(d, "overlay/mlx/backend/omarchy")
cmake = open(os.path.join(trees["branch"], "CMakeLists.txt")).read()
targets = re.findall(r"^\s*omarchy_shader\((\S+)\s+(shaders/qmm_vec\.comp)([^)]*)\)", cmake, re.M)
print("TARGETS", len(targets), flush=True)
same = diff = 0
for name, source, args in targets:
    digests = {}
    for tree, root in trees.items():
        out = os.path.join(work, f"{tree}-{name}.spv")
        r = subprocess.run(["glslc", "-O", "--target-env=vulkan1.3", *args.split(), os.path.join(root, source), "-o", out],
                           capture_output=True, text=True)
        digests[tree] = hashlib.sha256(open(out, "rb").read()).hexdigest()[:12] if r.returncode == 0 else "COMPILE-FAIL " + r.stderr[:120]
    verdict = "SAME" if digests["tag"] == digests["branch"] else "DIFF"
    same += verdict == "SAME"
    diff += verdict == "DIFF"
    print(f"SPV {verdict} {name} tag {digests['tag']} branch {digests['branch']}", flush=True)
print("SPV-SUMMARY same", same, "diff", diff, flush=True)
