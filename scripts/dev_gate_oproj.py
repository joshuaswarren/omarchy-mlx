#!/usr/bin/env python3
"""h13: end-to-end device-execution gate for the encoder rank-3 linear
(m375 k1024 n1024 uniform).

The 874/874 byte-parity suite is structurally blind to plane-permutation
bugs because the captured oracles are uniform-payload (every half is
identical), so any permutation produces the same bytes — that is the
mechanism that let the wrong oproj function ship green (rel_l2=1.37 vs
fp32 ref). This gate:

  1. Compiles a real-weight (m375, k1024, n1024) uniform linear via the
     mil-hwxc under test, with bias=0 and weights sampled from a
     deterministic rng. The emitter must stamp real bias halves
     (encodeLinearParity uniformBiasHalves parameter) and apply the
     64-plane inverse permutation (commit 67dfcf1).
  2. Executes on the device against real weights via libane-strict.
  3. Compares the device output against the fp32 reference across three
     independent rngs.
  4. Fails if any rng exceeds the rel_l2 budget (0.05; the fp16 W noise
     floor is ~0.012).

MUST be invoked on m1-test-host (where the aarch64 compiler lives) under the
flock -w 120 /tmp/m1-gpu.lock. Exits 0 on PASS, 1 on FAIL.

Gated to the (375, 1024, 1024) uniform geometry — the only one
CRT-measured on device. Smaller/different geometries stay on the
byte-parity gate until their own CRT runs.
"""
import ctypes, sys, hashlib, subprocess, json, os, struct, tempfile, math
from pathlib import Path
import numpy as np

LIBANE = "/var/tmp/island-reexport/libane-strict.so"
COMPILER = Path("$HOME/src/mil-hwx-compiler/build/mil-hwxc")
WORKDIR = Path("/tmp/dev-gate-oproj-local")

RELAY_BUDGET = 0.05
SEEDS = (11, 33, 57)


def make_real_oproj_mil(model_root: Path):
    rng = np.random.default_rng(0)
    W = rng.standard_normal((1024, 1024)).astype(np.float16)
    bias = np.zeros((1024,), dtype=np.float16)

    data_start = (64 + 4096 * 24 + 63) & ~63
    w_bytes = W.tobytes()
    b_bytes = bias.tobytes()
    total = data_start + len(w_bytes) + len(b_bytes)
    blob = bytearray(total)
    struct.pack_into('<II', blob, 0, 2, 2)
    struct.pack_into('<II', blob, 64, 0xDEADBEEF, 1)
    struct.pack_into('<QQ', blob, 64 + 8, len(w_bytes), data_start)
    struct.pack_into('<II', blob, 88, 0xDEADBEEF, 1)
    struct.pack_into('<QQ', blob, 88 + 8, len(b_bytes), data_start + len(w_bytes))
    blob[data_start:data_start + len(w_bytes)] = w_bytes
    blob[data_start + len(w_bytes):data_start + len(w_bytes) + len(b_bytes)] = b_bytes
    (model_root / 'weights.bin').write_bytes(bytes(blob))
    # Save W for the reference
    np.save(model_root / 'W_fp16.npy', W)
    np.save(model_root / 'bias_fp16.npy', bias)

    mil = '''program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {
    tensor<fp16, [1024, 1024]> w = const()[name = string("w"), val = tensor<fp16, [1024, 1024]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [1024]> b = const()[name = string("b"), val = tensor<fp16, [1024]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(88)))];
    tensor<fp16, [1, 375, 1024]> y = linear(weight = w, bias = b, x = x)[name = string("y")];
  } -> (y);
}
'''
    (model_root / 'test.mil').write_text(mil)
    return model_root / 'test.mil'


def run_on_device(anec_path: str, seed: int):
    """Execute the .anec on m1-test-host ANE under flock and return device output y."""
    py = f'''
import ctypes, sys
from pathlib import Path
import numpy as np
lib = ctypes.CDLL("{LIBANE}")
for n, r, a in (
    ("__ane_init", ctypes.c_void_p, [ctypes.c_char_p, ctypes.c_int]),
    ("__ane_free", None, [ctypes.c_void_p]),
    ("ane_exec", ctypes.c_int, [ctypes.c_void_p]),
    ("__ane_src_size", ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32]),
    ("__ane_dst_size", ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32]),
    ("__ane_send", None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]),
    ("__ane_read", None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]),
):
    f = getattr(lib, n); f.restype = r; f.argtypes = a
def run(anec, x16):
    nn = lib.__ane_init(str(anec).encode(), 0); assert nn
    try:
        s0, d0 = int(lib.__ane_src_size(nn, 0)), int(lib.__ane_dst_size(nn, 0))
        tile = np.zeros(s0, dtype=np.uint8)
        tile[0:x16.nbytes] = np.frombuffer(x16.tobytes(), dtype=np.uint8)
        lib.__ane_send(nn, ctypes.c_char_p(tile.ctypes.data), 0)
        assert lib.ane_exec(nn) == 0
        y = np.zeros(d0 // 2, dtype="<f2")
        lib.__ane_read(nn, ctypes.c_char_p(y.ctypes.data), 0)
        return y.astype(np.float32)
    finally:
        lib.__ane_free(nn)
seed = int(sys.argv[1])
rng = np.random.default_rng(seed)
x = rng.standard_normal((375, 1024)).astype(np.float16)
y = run("{anec_path}", x)[:384000].reshape(375, 1024)
np.save("/tmp/dev_gate_oproj_y_seed{{seed}}.npy".format(seed=seed), y)
'''
    pyfile = f"/tmp/dev_gate_oproj_run_seed{seed}.py"
    Path(pyfile).write_text(py)


def main():
    if not COMPILER.exists():
        print(f"FAIL: compiler not found at {COMPILER}")
        sys.exit(1)

    WORKDIR.mkdir(parents=True, exist_ok=True)
    model_root = WORKDIR / "model"
    model_root.mkdir(exist_ok=True)
    mil_path = make_real_oproj_mil(model_root)

    out_dir = WORKDIR / "out"
    if out_dir.exists():
        import shutil; shutil.rmtree(out_dir)
    r = subprocess.run(
        [str(COMPILER), "--mil", str(mil_path), "--model-root", str(model_root),
         "--target", "H13", "--format", "anec", "--output", str(out_dir)],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL: compile failed")
        print(r.stderr)
        sys.exit(1)
    local_anec = out_dir / "program-0.anec"
    if not local_anec.exists():
        print(f"FAIL: no .anec produced")
        sys.exit(1)

    # Load W for the fp32 reference
    W = np.load(model_root / 'W_fp16.npy').astype(np.float32)
    B = np.zeros((1024,), dtype=np.float32)

    # Write per-seed run scripts (no exec yet)
    for seed in SEEDS:
        run_on_device(str(local_anec), seed)
    # Run on device for each seed, under flock
    flock_cmd = (
        f"flock -w 300 /tmp/m1-gpu.lock -c '"
        + "; ".join([
            f"python3 /tmp/dev_gate_oproj_run_seed{s}.py {s}"
            for s in SEEDS
        ])
        + "'"
    )
    flock_proc = subprocess.run(flock_cmd, shell=True, capture_output=True, text=True, timeout=600)
    if flock_proc.returncode != 0:
        print("device run failed")
        print("stdout:", flock_proc.stdout[:500])
        print("stderr:", flock_proc.stderr[:500])
        sys.exit(2)

    results = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((375, 1024)).astype(np.float16).astype(np.float32)
        ref = x @ W.T + B
        yfile = f"/tmp/dev_gate_oproj_y_seed{seed}.npy"
        if not Path(yfile).exists():
            print(f"FAIL: device output missing for seed={seed}")
            sys.exit(1)
        y = np.load(yfile).reshape(375, 1024)
        rel = float(np.linalg.norm(y - ref) / np.linalg.norm(ref))
        results.append({"seed": seed, "rel_l2": rel})

    worst = max(r["rel_l2"] for r in results)
    summary = {
        "compiler": str(COMPILER),
        "compiler_commit_required": "67dfcf1 + 5bd7ad5",
        "anec": str(local_anec),
        "anec_sha256": hashlib.sha256(local_anec.read_bytes()).hexdigest(),
        "rngs": results,
        "worst_rel_l2": worst,
        "budget": RELAY_BUDGET,
        "pass": worst <= RELAY_BUDGET,
    }
    print(json.dumps(summary, indent=2))
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()