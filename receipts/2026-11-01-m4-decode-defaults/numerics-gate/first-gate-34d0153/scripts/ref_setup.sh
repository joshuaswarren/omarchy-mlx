#!/bin/bash
# Reference venv for docs/numerics-gate.md: stock upstream MLX 0.32.3 (the
# mlx.lock pin) CPU backend from PyPI plus stock mlx-lm 0.31.3, in a fresh venv
# used only for the reference. Every downloaded wheel is checked against the
# sha256 PyPI publishes before it is installed; installed files are checked
# against each distribution's RECORD afterwards.
set -euo pipefail
U=$HOME/gpu/upr; W=$U/ref-wheels; V=$U/venv-ref
mkdir -p "$W"
fetch() {  # fetch <project> <version> <filename>
  python3 - "$1" "$2" "$3" "$W" <<'PY'
import hashlib, json, sys, urllib.request
proj, ver, name, out = sys.argv[1:]
meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{proj}/{ver}/json", timeout=30))
f = [u for u in meta["urls"] if u["filename"] == name]
assert len(f) == 1, (proj, ver, name)
want = f[0]["digests"]["sha256"]
data = urllib.request.urlopen(f[0]["url"], timeout=120).read()
got = hashlib.sha256(data).hexdigest()
print(f"{name} size={len(data)} pypi_sha256={want} downloaded_sha256={got} {'MATCH' if got == want else 'MISMATCH'}")
if got != want:
    sys.exit(1)
open(f"{out}/{name}", "wb").write(data)
PY
}
fetch mlx 0.32.3 mlx-0.32.3-cp314-cp314-manylinux_2_35_aarch64.whl
fetch mlx-cpu 0.32.3 mlx_cpu-0.32.3-py3-none-manylinux_2_35_aarch64.whl
fetch mlx-lm 0.31.3 mlx_lm-0.31.3-py3-none-any.whl
rm -rf "${V:?}"
python3 -m venv "$V"
"$V/bin/pip" install -q --no-deps "$W"/mlx-0.32.3-*.whl "$W"/mlx_cpu-0.32.3-*.whl "$W"/mlx_lm-0.31.3-*.whl
"$V/bin/pip" install -q -r $U/probes/runtime-reqs.txt
"$V/bin/python" - <<'PY'
import base64, hashlib, importlib.metadata as md, os
for name in ("mlx", "mlx-cpu", "mlx-lm"):
    d = md.distribution(name)
    ok = bad = 0
    for f in d.files:
        if f.hash is None:
            continue
        p = f.locate()
        h = base64.urlsafe_b64encode(hashlib.new(f.hash.mode, open(p, "rb").read()).digest()).rstrip(b"=").decode()
        if h == f.hash.value:
            ok += 1
        else:
            bad += 1
            print("RECORD MISMATCH", name, f)
    print(f"RECORD {name} {d.version}: {ok} files match, {bad} differ")
    assert bad == 0
names = sorted(d.metadata["Name"] for d in md.distributions())
assert not any("omarchy" in n.lower() for n in names), names
print("no omarchy distribution in this venv")
import mlx.core as mx
print("mlx", mx.__version__, "default device", mx.default_device(), "file", mx.__file__.replace(os.path.expanduser("~"), "~"))
x = mx.random.normal((2, 8, 4, 64))
w = mx.ones((64,))
y = mx.fast.rms_norm(x, w, 1e-5); r = mx.fast.rope(x, 64, traditional=False, base=10000.0, scale=1.0, offset=3)
s = mx.fast.scaled_dot_product_attention(x, x, x, scale=0.125)
mx.eval(y, r, s)
print("cpu rms_norm, rope, sdpa ran:", y.shape, r.shape, s.shape)
PY
echo REF_OK
