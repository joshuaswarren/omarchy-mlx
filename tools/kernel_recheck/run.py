"""Kernel recheck runner.

Dev box (no mlx import):
    python3 -m tools.kernel_recheck.run --reference-only [--kernel NAME]
Builds inputs, runs the NumPy references, prints one DRYRUN line per kernel.

GPU host (jw16, under the wheel python):
    python3 -m tools.kernel_recheck.run --gpu [--kernel NAME]
Runs every kernel in its own subprocess (`--single NAME`), classifies the
outcome, prints `KERNEL <name> STATUS pass|fail|refused <first error line>
maxdiff <x>` and exits nonzero if any kernel produced a wrong value.

Status classes:
  pass      dispatched and within the pre-declared tolerance of the reference
  refused   the translator's named refusal ("unsupported" / "[omarchy]")
  fail      crash (process abort / Vulkan / GLSL compile failure) or timeout
  wrong     dispatched but outside tolerance (worst class; forces exit 3)
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import defs


def source_text(spec):
    path = Path(__file__).parent / "sources" / defs.UPSTREAM[spec["name"]]
    text = path.read_text()
    fmt = spec.get("source_format")
    if fmt:
        text = text.format(**fmt)
    return text


def build_inputs(spec):
    arrays = spec["builder"]()
    names = spec["inputs"]
    return {n: arrays[n] for n in names}


def check(name, spec, inputs, refs, outs_np):
    """Compare kernel outputs against references; returns (ok, maxdiff, detail)."""
    worst = 0.0
    for i, (ref, out, tol) in enumerate(zip(refs, outs_np, spec["tol"])):
        out = np.asarray(out)
        if ref.shape != out.shape:
            return False, float("nan"), f"shape {list(out.shape)} != ref {list(ref.shape)}"
        if tol[0] == "exact":
            bad = ~np.isclose(out.astype(np.float64), ref.astype(np.float64),
                              rtol=0, atol=0, equal_nan=True)
            diff = float(np.max(np.abs(out.astype(np.float64) - ref.astype(np.float64)))) if out.size else 0.0
        else:
            rtol = tol[1]
            atol = tol[2] if len(tol) > 2 else 0.0
            err = np.abs(out.astype(np.float64) - ref)
            limit = atol + rtol * np.abs(ref)
            bad = err > limit
            diff = float(np.max(err / np.maximum(np.abs(ref), 1e-300))) if out.size else 0.0
        worst = max(worst, diff if diff == diff else 1e308)
        if bad.any():
            j = int(np.argmax(np.abs(out.astype(np.float64) - ref)))
            return False, worst, f"out[{i}] bad {int(bad.sum())}/{bad.size} at {np.unravel_index(j, out.shape)}: got {out.flat[j]!r} want {ref.flat[j]!r}"
    return True, worst, ""


# ---------------------------------------------------------------- dry run
def reference_only(only=None):
    ok = True
    for spec in defs.SPECS:
        name = spec["name"]
        if only and name != only:
            continue
        inputs = build_inputs(spec)
        try:
            refs = spec["ref"](inputs)
        except Exception as error:  # noqa: BLE001
            print(f"KERNEL {name} dryrun ref-error {type(error).__name__}: {error}")
            ok = False
            continue
        shapes = ";".join("x".join(map(str, r.shape)) for r in refs)
        kinds = ";".join(("int" if r.dtype.kind in "ui" else "f64") for r in refs)
        bad_shape = any(list(r.shape) != list(s) for r, s in zip(refs, spec["out_shapes"]))
        status = "ref-badshape" if bad_shape else "ref-ok"
        # +-inf is a legitimate reference value (mask kernels); NaN is not
        finite = all(not np.isnan(r.astype(np.float64)).any() for r in refs
                     if r.dtype.kind == "f")
        if bad_shape or not finite:
            status = "ref-bad"
            ok = False
        print(f"KERNEL {name} dryrun {status} predict={defs.PREDICT[name]} "
              f"outs=({shapes}) kinds=({kinds}) maxdiff 0")
    return 0 if ok else 1


# ---------------------------------------------------------------- gpu child
def single(name):
    """One kernel on the GPU. Prints 'RESULT <status> maxdiff <x> <detail>'
    as the LAST stdout line; exit 0 pass, 3 wrong, 1 anything else.
    Exceptions print a one-line 'CHILDERR: [Type] message' so the parent can
    classify translator refusals without scraping tracebacks."""
    try:
        return _single(name)
    except SystemExit:
        raise
    except BaseException as error:  # noqa: BLE001 - the parent classifies
        text = str(error).strip()
        first = text.splitlines()[0] if text else type(error).__name__
        print(f"CHILDERR: [{type(error).__name__}] {first[:900]}")
        return 1


def _single(name):
    import mlx.core as mx  # deferred: not needed for --reference-only

    spec = defs.get_spec(name)
    inputs = build_inputs(spec)

    kwargs = {}
    if spec.get("header"):
        kwargs["header"] = (Path(__file__).parent / "sources" / spec["header"]).read_text()
    if spec.get("atomic_outputs"):
        kwargs["atomic_outputs"] = True
    if spec.get("ensure_row_contiguous") is False:
        kwargs["ensure_row_contiguous"] = False

    kernel = mx.fast.metal_kernel(
        name=name,
        input_names=list(spec["inputs"]),
        output_names=list(spec["outputs"]),
        source=source_text(spec),
        **kwargs,
    )

    dtypes = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16,
              "uint32": mx.uint32, "int32": mx.int32, "uint8": mx.uint8,
              "uint16": mx.uint16, "int8": mx.int8}
    mx_arrays = []
    for n, dt_name in zip(spec["inputs"], defs.IN_DTYPES[name]):
        a = inputs[n]
        if dt_name == "bfloat16":
            a32 = np.ascontiguousarray(a, dtype=np.float32)
            mx_arrays.append(mx.array(a32).astype(mx.bfloat16))
        else:
            mx_arrays.append(mx.array(a))
    template = []
    for key, value in spec.get("template", []):
        template.append((key, dtypes[value] if isinstance(value, str) and value in dtypes else value))

    call_kwargs = dict(
        inputs=mx_arrays,
        template=template,
        grid=spec["grid"],
        threadgroup=spec["threadgroup"],
        output_shapes=[tuple(s) for s in spec["out_shapes"]],
        output_dtypes=[dtypes[d] for d in spec["out_dtypes"]],
        stream=mx.gpu,
    )
    if spec.get("init_value") is not None:
        call_kwargs["init_value"] = spec["init_value"]

    outs = kernel(**call_kwargs)
    for o in outs:
        mx.eval(o)

    refs = spec["ref"](inputs)
    outs_np = [np.array(o.astype(mx.float32)) if o.dtype == mx.bfloat16
               else np.array(o) for o in outs]
    ok, worst, detail = check(name, spec, inputs, refs, outs_np)
    if ok:
        print(f"RESULT pass maxdiff {worst:.6g}")
        return 0
    print(f"RESULT wrong maxdiff {worst:.6g} {detail}")
    return 3


# ---------------------------------------------------------------- gpu parent
def gpu(only=None, timeout=180):
    rows = []
    any_wrong = False
    stderr_dir = Path(os.environ.get("KERNEL_RECHECK_STDERR", "stderr"))
    stderr_dir.mkdir(parents=True, exist_ok=True)
    for spec in defs.SPECS:
        name = spec["name"]
        if only and name != only:
            continue
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "tools.kernel_recheck.run", "--single", name],
                capture_output=True, text=True, timeout=timeout)
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as expired:
            rc = None
            out = expired.stdout or ""
            err = (expired.stderr or "") + f"\n[timeout after {timeout}s]"
        (stderr_dir / f"{name}.stderr").write_text(
            f"# rc={rc}\n--- stdout ---\n{out}\n--- stderr ---\n{err}")
        elapsed = time.time() - start
        predict = defs.PREDICT[name]
        result_line = next((l for l in out.splitlines() if l.startswith("RESULT")), "")
        child_line = next((l for l in out.splitlines() if l.startswith("CHILDERR: ")), "")
        if rc == 0 and result_line:
            status = "pass"
            maxdiff = result_line.split()[3] if len(result_line.split()) > 3 else "n/a"
            first = ""
        elif rc == 3 and result_line:
            status = "wrong"
            maxdiff = result_line.split()[3] if len(result_line.split()) > 3 else "n/a"
            first = result_line.split(" ", 4)[-1] if len(result_line.split()) > 4 else ""
        else:
            text = (child_line or err or out).strip()
            first = text.splitlines()[0] if text else f"exit {rc}"
            if child_line.startswith("CHILDERR: [omarchy]") or "unsupported" in (child_line or err):
                status = "refused"
            else:
                status = "fail"
            first = first[:200]
            maxdiff = "n/a"
        rows.append((name, status, predict))
        if status == "wrong":
            any_wrong = True
        tail = first if status in ("refused", "fail", "wrong") else ""
        print(f"KERNEL {name} {status} {tail} maxdiff {maxdiff} [{elapsed:.1f}s] "
              f"predict={predict}")
    passed = sum(1 for r in rows if r[1] == "pass")
    wrong = sum(1 for r in rows if r[1] == "wrong")
    refused = sum(1 for r in rows if r[1] == "refused")
    failed = sum(1 for r in rows if r[1] == "fail")
    hits = sum(1 for n, s, p in rows if s == p or (p == "crash" and s == "fail"))
    print(f"SUMMARY pass={passed} wrong={wrong} refused={refused} fail={failed} "
          f"total={len(rows)} prediction_hits={hits}")
    return 3 if any_wrong else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--single")
    parser.add_argument("--kernel")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.single:
        sys.exit(single(args.single))
    if args.reference_only:
        sys.exit(reference_only(args.kernel))
    if args.gpu:
        sys.exit(gpu(args.kernel, args.timeout))
    parser.error("pick --reference-only, --gpu, or --single")


if __name__ == "__main__":
    main()
