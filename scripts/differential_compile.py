#!/usr/bin/env python3
"""Differential harness: compiled vs eager on intermediate values, first divergence.

The compiled path on Honeykrisp returns silently wrong values at commit
ff4b05a and later: a Qwen2.5-0.5B-Instruct-4bit greedy decode emitted garbage
with compilation enabled while MLX_DISABLE_COMPILE=1 was correct, at normal
speed and exit code 0.  The C++ batteries pass on llvmpipe and cannot see
this class of defect.  This harness narrows it:

1. Graph mode: run a synthetic computation twice on the same device with
   identical inputs and seed - every compile unit wrapped in mx.compile
   versus fully eager - and compare every unit boundary bit-for-bit.
2. Model mode: run the Qwen forward pass layer by layer and find the first
   diverging layer, then descend into that layer and name the first
   diverging operation inside it (linears, rope, sdpa, swiglu, norms).
3. Real-path mode: the exact native compile structure mlx-lm applies
   (per-fragment shapeless compiles such as mlx_lm.models.qwen2.swiglu),
   run as two env-isolated subprocesses over the decode loop.
4. Shrink mode: emit the smallest standalone reproduction of a diverging
   operation: an inputs .npz plus a repro script that rebuilds only that
   operation and compares compiled vs eager.

Every recorded value sits on a compile-unit boundary, so it is evaluable in
both passes.  Mid-tape arrays are not evaluable by design; the descent adds
finer unit boundaries instead of trying to read tape internals.

Equality policy: BITWISE.  Both passes run the same primitives on the same
device with identical inputs, so identical bits is the only legitimate
outcome.  Any tolerance would hide exactly the corruption this harness
exists to find.  `--atol` exists only for explicit cross-path numerics
debugging and prints a loud warning; the default is exact.

The mechanism-level probe lives in scripts/probe_tape_eager.py.

llvmpipe limitation: llvmpipe executes submissions synchronously inside
vkQueueSubmit, so it cannot expose asynchronous or cross-submission
corruption.  A clean llvmpipe run is the start of verification, not proof.
See docs/differential-harness.md.

Exit codes: 0 match, 3 divergence found or injected divergence detected,
2 usage or environment error.

Self-test (no mlx, no GPU): python3 differential_compile.py --self-test
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

EXIT_MATCH = 0
EXIT_DIVERGE = 3
EXIT_ERROR = 2

# Absolute path of this script's directory, baked into generated repro.py
# so `from differential_compile import ...` resolves from any cwd.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Comparator: numpy-only so the logic is testable without mlx (--self-test)
# --------------------------------------------------------------------------

def unsigned_view(a):
    """Bitwise view of a numpy array: same itemsize, unsigned integer."""
    codes = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
    return a.view(codes[a.dtype.itemsize])


def bitwise_equal(a, b):
    """Exact equality on raw bits, so NaN payloads compare exactly too."""
    if a.shape != b.shape or a.dtype.itemsize != b.dtype.itemsize:
        return False
    return bool(np.array_equal(unsigned_view(a), unsigned_view(b)))


def first_diff_index(a, b):
    """Flat index of the first differing element in C order, else -1."""
    neq = unsigned_view(a) != unsigned_view(b)
    flat = neq.reshape(-1)
    return int(np.argmax(flat)) if flat.any() else -1


def value_description(a, idx):
    """The float reading and the raw bits at one flat index."""
    flat = a.reshape(-1)
    v = flat[idx]
    shown = repr(float(v)) if a.dtype.kind == "f" else str(v)
    bits = "0x" + unsigned_view(flat[idx: idx + 1]).tobytes().hex()
    return {"value": shown, "bits": bits}


def compare_capture(a_caps, b_caps, atol=None):
    """Compare two captures of the same boundary.  Returns divergence or None.

    atol=None compares bitwise.  With atol, a divergence is any element pair
    outside the tolerance - loud, explicit, never the default.
    """
    if len(a_caps) != len(b_caps):
        return {"kind": "arity", "detail": f"{len(a_caps)} vs {len(b_caps)} outputs"}
    for j, (ca, cb) in enumerate(zip(a_caps, b_caps)):
        na, nb = ca["np"], cb["np"]
        if na.shape != nb.shape:
            return {"kind": "shape", "output": j,
                    "detail": f"{list(na.shape)} vs {list(nb.shape)}"}
        if ca["dtype"] != cb["dtype"]:
            return {"kind": "dtype", "output": j,
                    "detail": f"{ca['dtype']} vs {cb['dtype']}"}
        if atol is None:
            if not bitwise_equal(na, nb):
                idx = first_diff_index(na, nb)
                return {"kind": "value", "output": j, "index": idx,
                        "eager": value_description(na, idx),
                        "compiled": value_description(nb, idx)}
        else:
            if not np.allclose(na.astype(np.float64), nb.astype(np.float64),
                               atol=atol, rtol=0):
                idx = int(np.argmax(np.abs(na.astype(np.float64) - nb.astype(np.float64))
                                    > atol))
                return {"kind": "value(tolerance)", "output": j, "index": idx,
                        "eager": value_description(na, idx),
                        "compiled": value_description(nb, idx)}
    return None


# --------------------------------------------------------------------------
# MLX value capture
# --------------------------------------------------------------------------

def mlx_to_numpy(a):
    """Host copy of an mlx array as numpy, bitwise-preserving.

    bf16 has no numpy dtype, so it travels as its exact uint16 bits; the
    dtype name keeps the interpretation.
    """
    import mlx.core as mx

    t = str(a.dtype)
    if t.startswith("mlx.core."):
        t = t.split(".")[-1]
    if t == "bfloat16":
        return np.asarray(a.view(mx.uint16)), t
    return np.asarray(a), t


def capture_values(arrays):
    """mx.eval then host-copy a list of mlx arrays."""
    import mlx.core as mx

    for a in arrays:
        mx.eval(a)
    caps = []
    for a in arrays:
        n, t = mlx_to_numpy(a)
        caps.append({"np": n, "dtype": t, "shape": list(n.shape)})
    return caps


# --------------------------------------------------------------------------
# Units and the differential runner
# --------------------------------------------------------------------------

class Unit:
    """One compile boundary: a callable applied to the running values.

    `attr_path` locates the callable inside a loaded model (model mode).
    `repro_import` is python source that rebuilds the callable standalone.
    `builder` is python source `def make_fn(args, inputs):` rebuilding a graph unit.
    """

    def __init__(self, label, fn, attr_path=None, repro_import=None, builder=None):
        self.label = label
        self.fn = fn
        self.attr_path = attr_path
        self.repro_import = repro_import
        self.builder = builder
        self._compiled = None

    def call(self, args, compiled):
        if not compiled:
            return self.fn(*args)
        if self._compiled is None:
            import mlx.core as mx

            self._compiled = mx.compile(self.fn)
        return self._compiled(*args)


class Divergence:
    def __init__(self, step, label, detail, inputs, unit=None):
        self.step = step
        self.label = label
        self.detail = detail
        self.inputs = inputs
        self.unit = unit
        self.args_arrays = None   # the real mlx input arrays of the diverging unit

    def render(self):
        d = self.detail
        lines = [f"FIRST DIVERGENCE  step={self.step}  operation='{self.label}'",
                 f"  kind: {d['kind']}"]
        if "shapes" in d:
            lines.append(f"  shapes: {d['shapes']}  dtypes: {d['dtypes']}")
        if d["kind"].startswith("value"):
            lines.append(f"  first differing flat index: {d['index']}")
            lines.append(f"  eager   : {d['eager']['value']} (bits {d['eager']['bits']})")
            lines.append(f"  compiled: {d['compiled']['value']} (bits {d['compiled']['bits']})")
        elif "detail" in d:
            lines.append(f"  detail: {d['detail']}")
        if self.inputs:
            lines.append(f"  inputs: {self.inputs}")
        return "\n".join(lines)


def diff_records(rec_e, rec_c, atol=None, step="?"):
    """Align two record lists position-by-position; return Divergence or None.

    A record is {"label", "args", "outs"}; outs are captured on demand.
    """
    if len(rec_e) != len(rec_c):
        labels_e = [r["label"] for r in rec_e]
        labels_c = [r["label"] for r in rec_c]
        return Divergence(step, "record-sequence",
                          {"kind": "structure",
                           "detail": f"record sequences differ: {len(rec_e)} vs {len(rec_c)}; "
                                     f"eager={labels_e} compiled={labels_c}"}, None)
    for re_, rc in zip(rec_e, rec_c):
        if re_["label"] != rc["label"]:
            return Divergence(step, "record-sequence",
                              {"kind": "structure",
                               "detail": f"order diverged: {re_['label']} vs {rc['label']}"}, None)
        ea = capture_values(list(re_["outs"]))
        ca = capture_values(list(rc["outs"]))
        d = compare_capture(ea, ca, atol=atol)
        if d is not None:
            d["shapes"] = [c["shape"] for c in ea]
            d["dtypes"] = [c["dtype"] for c in ea]
            ins = [{"shape": list(np.shape(a)),
                    "dtype": mlx_to_numpy(a)[1] if hasattr(a, "dtype") else "python"}
                   for a in re_["args"]]
            div = Divergence(step, re_["label"], d, ins, unit=re_["unit"])
            div.args_arrays = list(re_["args"])
            return div
    return None


# --------------------------------------------------------------------------
# Graph mode: synthetic computations for harness verification and op narrowing
# --------------------------------------------------------------------------

def graph_chain(seed, size, dtype_name):
    """Mixed fusable/eager chain with a one-element-output reduction."""
    import mlx.core as mx

    dt = getattr(mx, dtype_name)
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((size, size)).astype(np.float32), dtype=dt)
    w = mx.array(rng.standard_normal((size, size)).astype(np.float32), dtype=dt)
    units = [
        Unit("sin-chain", lambda a: mx.sin(a) * 2.0 + a,
             builder="def make_fn(args, inputs):\n"
                     "    return lambda a: mx.sin(a) * 2.0 + a\n"),
        Unit("where-exp", lambda a: mx.where(a > 0, mx.exp(-a * a), mx.cos(a)),
             builder="def make_fn(args, inputs):\n"
                     "    return lambda a: mx.where(a > 0, mx.exp(-a * a), mx.cos(a))\n"),
        Unit("matmul", lambda a, w: mx.matmul(a, w) * 0.01,
             builder="def make_fn(args, inputs):\n"
                     "    return lambda a, w: mx.matmul(a, w) * 0.01\n"),
        Unit("sum-to-one", lambda a: mx.sum(a, axis=0, keepdims=True),
             builder="def make_fn(args, inputs):\n"
                     "    return lambda a: mx.sum(a, axis=0, keepdims=True)\n"),
        Unit("broadcast-add", lambda a: a + 1.0,
             builder="def make_fn(args, inputs):\n"
                     "    return lambda a: a + 1.0\n"),
    ]
    fixed = [x, w]
    arg_specs = [[0], [None], [None, 1], [None], [None]]
    return units, fixed, arg_specs


def graph_swiglu(seed, size, dtype_name):
    """The mlx-lm swiglu fragment shape: silu(gate) * up plus eager glue."""
    import mlx.core as mx

    dt = getattr(mx, dtype_name)
    rng = np.random.default_rng(seed)
    g = mx.array(rng.standard_normal((size, size)).astype(np.float32), dtype=dt)
    u = mx.array(rng.standard_normal((size, size)).astype(np.float32), dtype=dt)
    silu = lambda a: a * (1.0 / (1.0 + mx.exp(-a)))
    units = [
        Unit("silu(gate)", silu,
             builder="def make_fn(args, inputs):\n"
                     "    return lambda a: a * (1.0 / (1.0 + mx.exp(-a)))\n"),
        Unit("mul(up)", lambda s, b: s * b,
             builder="def make_fn(args, inputs):\n    return lambda s, b: s * b\n"),
        Unit("tanh-glue", lambda m: mx.where(m > 0, mx.tanh(m) + 1.0, mx.erf(m)),
             builder="def make_fn(args, inputs):\n"
                     "    return lambda m: mx.where(m > 0, mx.tanh(m) + 1.0, mx.erf(m))\n"),
    ]
    fixed = [g, u]
    arg_specs = [[0], [None, 1], [None]]
    return units, fixed, arg_specs


def resolve_args(spec, prev_outs, fixed):
    """Spec slots: int = fixed/previous output index consumed as-is,
    None = all outputs of the previous unit."""
    row = []
    for s in spec:
        if s is None:
            row.extend(prev_outs[-1])
        else:
            row.append(fixed[s])
    return row


def run_graph(units, fixed, arg_specs, compiled, inject=None):
    prev_outs = []
    recs = []
    for i, (u, spec) in enumerate(zip(units, arg_specs)):
        args = resolve_args(spec, prev_outs, fixed)
        out = u.call(args, compiled)
        out = out if isinstance(out, tuple) else (out,)
        if inject and inject[0] == i and compiled:
            _, oi, addend = inject
            out = list(out)
            out[oi] = out[oi] + addend   # simulated tape corruption, propagates
            out = tuple(out)
        recs.append({"label": u.label, "unit": u, "args": args, "outs": out})
        prev_outs.append(out)
    return recs


# --------------------------------------------------------------------------
# Model mode: Qwen layer-by-layer, then operations inside the diverging layer
# --------------------------------------------------------------------------

class Recorder:
    """Collects boundary records; `context` names the enclosing layer."""

    def __init__(self):
        self.records = []
        self.context = ""

    def add(self, label, args, outs, unit=None):
        self.records.append({"label": (self.context + "." if self.context else "") + label,
                             "args": args, "outs": outs, "unit": unit})


class Wrap:
    """Record-and-optionally-compile replacement for a model callable.

    Unrecorded attribute access passes through to the wrapped module, so
    embed_tokens.as_linear keeps working under the wrap.
    """

    def __init__(self, fn, label, rec, compiled, ctx=False, module_fn=False):
        self._fn = fn
        self._label = label
        self._rec = rec
        self._compiled = compiled
        self._ctx = ctx
        self._module_fn = module_fn
        self._compiled_fn = None

    def __call__(self, *args, **kwargs):
        if self._compiled and self._compiled_fn is None:
            import mlx.core as mx

            self._compiled_fn = mx.compile(self._fn)
        fn = self._compiled_fn if self._compiled else self._fn
        if self._ctx:
            self._rec.context = self._label
        try:
            out = fn(*args, **kwargs)
            outs = out if isinstance(out, tuple) else (out,)
            unit = Unit(self._label, self._fn)
            if self._module_fn:
                unit.repro_import = (
                    "def make_fn(args, inputs):\n"
                    "    from mlx_lm.models import qwen2\n"
                    f"    return qwen2.{self._label.split('.')[-1]}\n"
                )
            else:
                ctx = self._rec.context
                unit.attr_path = (ctx + "." if ctx else "") + self._label
            self._rec.add(self._label, args, outs, unit=unit)
        finally:
            if self._ctx:
                self._rec.context = ""
        return out

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_fn"), name)


class ModulePatch:
    """Temporary replacement of a module-level function (e.g. qwen2.swiglu).

    The label carries the recorder's current layer context, so patched
    fragment calls are named per layer.
    """
    def __init__(self, module, name, rec, compiled, module_fn=False):
        self.module = module
        self.name = name
        self.rec = rec
        self.orig = getattr(module, name)

    def __enter__(self):
        setattr(self.module, self.name, self.wrap)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.orig)
        return False



def resolve_path(model, path):
    """'model.layers.3.self_attn.q_proj' -> the object."""
    obj = model
    for part in path.replace("[", ".").replace("]", "").split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def parent_and_name(model, path):
    parts = path.replace("[", ".").replace("]", "").split(".")
    return resolve_path(model, ".".join(parts[:-1])), parts[-1]


def install_model_wraps(model, rec, compiled, layer_only=None, internals=False):
    """Wrap model callables per granularity; returns an undo callable.

    Granularity "layer": every layer whole, plus embed/norm.
    Granularity "internals": unchanged except inside layer_only, where the
    attention/mlp sub-callables and the module-level swiglu/sdpa fragments
    are wrapped too.  The model's own code runs unchanged throughout.
    """
    import mlx_lm.models.qwen2 as q2

    undo = []

    def wrap_attr(path, label):
        parent, name = parent_and_name(model, path)
        orig = getattr(parent, name)
        setattr(parent, name, Wrap(orig, label, rec, compiled))
        undo.append(lambda: setattr(parent, name, orig))

    wrap_attr("model.embed_tokens", "model.embed_tokens")
    wrap_attr("model.norm", "model.norm")
    layers = resolve_path(model, "model.layers")
    for i, layer in enumerate(layers):
        if layer_only is None or i == layer_only:
            parent = layers
            orig = layer
            w = Wrap(orig, f"layers[{i}]", rec, compiled, ctx=True)
            parent[i] = w
            undo.append(lambda p=parent, j=i, o=orig: p.__setitem__(j, o))
        if internals and i == layer_only:
            for sub, leaves in (
                ("self_attn", ["q_proj", "k_proj", "v_proj", "o_proj", "rope"]),
                ("mlp", ["gate_proj", "up_proj", "down_proj"]),
            ):
                base = f"model.layers.{i}.{sub}"
                wrap_attr(base, f"{sub}")
                for leaf in leaves:
                    if getattr(resolve_path(model, base), leaf, None) is not None:
                        wrap_attr(f"{base}.{leaf}", f"{sub}.{leaf}")
            for ln in ("input_layernorm", "post_attention_layernorm"):
                wrap_attr(f"model.layers.{i}.{ln}", ln)

    # Module-level fragments: the actual compiled units of a native mlx-lm run.
    patches = [
        ModulePatch(q2, "swiglu", rec, compiled, module_fn=True),
        ModulePatch(q2, "scaled_dot_product_attention", rec, compiled, module_fn=True),
    ]

    def restore():
        for p in patches:
            p.__exit__()
        for u in undo:
            u()

    return restore


def run_model_pass(model, ids, compiled, rec, layer_only=None, internals=False):
    """One full-recompute forward through the model's own __call__ path.

    cache=None keeps every step a pure function of the token prefix, so a
    layer output is an evaluable function output under both settings.
    """
    import mlx.core as mx

    restore = install_model_wraps(model, rec, compiled, layer_only, internals)
    try:
        logits = model(mx.array([ids]), cache=None)
        mx.eval(logits)
        return logits
    finally:
        restore()


def load_qwen(model_path):
    try:
        from mlx_lm.utils import load

        return load(model_path)
    except ImportError:
        from mlx_lm.utils import load_model

        return load_model(model_path)


def model_ladder(model, tok, args):
    """Layer pass over the decode loop, then op descent into the first bad layer."""
    import mlx.core as mx

    if getattr(args, "no_chat_template", False):
        ids = tok.encode(args.prompt)
    else:
        try:
            ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                          add_generation_prompt=True)
        except Exception:
            ids = tok.encode(args.prompt)
    ids = list(ids)
    mx.random.seed(args.seed)

    cur = list(ids)
    first = None
    for step in range(args.steps + 1):
        rec_e = Recorder()
        rec_c = Recorder()
        mx.disable_compile()
        run_model_pass(model, cur, compiled=False, rec=rec_e)
        mx.enable_compile()
        run_model_pass(model, cur, compiled=True, rec=rec_c)
        mx.disable_compile()
        div = diff_records(rec_e.records, rec_c.records, atol=args.atol, step=step)
        if div is not None:
            first = div
            break
        logits = rec_e.records[-1]["outs"][0]
        cur.append(int(mx.argmax(logits)))
    mx.enable_compile()

    if first is None:
        return None, None

    print(first.render())
    # Op descent inside the first diverging layer.
    m = first.label
    if "layers[" not in m:
        return first, None
    layer_idx = int(m.split("layers[")[1].split("]")[0])
    rec_e = Recorder()
    rec_c = Recorder()
    mx.disable_compile()
    run_model_pass(model, cur, compiled=False, rec=rec_e,
                   layer_only=layer_idx, internals=True)
    mx.enable_compile()
    run_model_pass(model, cur, compiled=True, rec=rec_c,
                   layer_only=layer_idx, internals=True)
    mx.disable_compile()
    op_div = diff_records(rec_e.records, rec_c.records, atol=args.atol,
                          step=f"{first.step} descent")
    mx.enable_compile()
    if op_div is not None:
        print(op_div.render())
        return first, op_div
    print(f"op descent: every wrapped callable inside layers[{layer_idx}] matches bitwise; "
          "the divergence involves non-wrapped glue (residual adds, reshape/transpose chains, "
          "as_linear head) or the whole-layer tape structure")
    return first, None


# --------------------------------------------------------------------------
# Shrink mode
# --------------------------------------------------------------------------

def standalone_diverges(unit, inputs, atol=None):
    """Re-run one unit alone: eager vs compiled on the given inputs."""
    e = unit.call(list(inputs), compiled=False)
    e = e if isinstance(e, tuple) else (e,)
    ea = capture_values(list(e))
    c = unit.call(list(inputs), compiled=True)
    c = c if isinstance(c, tuple) else (c,)
    ca = capture_values(list(c))
    return compare_capture(ea, ca, atol=atol) is not None


def shrink_inputs(unit, inputs, atol=None, max_tries=24):
    """Halve axis 0 of the largest input while the standalone divergence lives."""
    kept = list(inputs)
    for _ in range(max_tries):
        biggest = max(range(len(kept)), key=lambda i: kept[i].size)
        a = kept[biggest]
        if a.shape[0] <= 1:
            break
        cand = list(kept)
        cand[biggest] = a[: a.shape[0] // 2]
        try:
            if not standalone_diverges(unit, cand, atol=atol):
                break
        except Exception:
            break
        kept = cand
    return kept


def save_inputs(path, inputs):
    arrays = {}
    for i, a in enumerate(inputs):
        n, t = mlx_to_numpy(a) if hasattr(a, "dtype") else (np.asarray(a), "float32")
        arrays[f"in{i}"] = n
        arrays[f"in{i}_dtype"] = np.array(t)
    np.savez(path, **arrays)


REPRO_TEMPLATE = '''#!/usr/bin/env python3
"""Standalone shrink reproduction generated by differential_compile.py.

Unit: {label}
Rebuilds only this operation and its saved inputs, runs it eager and
compiled on the same device, and compares bitwise.

Run (compile enabled - do NOT set MLX_DISABLE_COMPILE):
    unset MLX_DISABLE_COMPILE
    {py} {repro} [--model {model_hint}]
Requires the mlx-omarchy wheel built at the commit under test.
Exit 3 on divergence, 0 on match.
"""
import argparse
import os
import sys

import numpy as np

import mlx.core as mx

sys.path.insert(0, {scripts_dir!r})
from differential_compile import Unit, standalone_diverges

{rebuild}

def load_inputs():
    base = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(base, "repro.npz"), allow_pickle=False)
    n_in = len([k for k in data.files if k.startswith("in") and not k.endswith("_dtype")])
    return [mx.array(data[f"in{{i}}"], dtype=getattr(mx, str(data[f"in{{i}}_dtype"])))
            for i in range(n_in)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default={model_hint!r})
    args = ap.parse_args()
    inputs = load_inputs()
    fn = make_fn(args, inputs)
    div = standalone_diverges(Unit({label!r}, fn), inputs)
    if div:
        print("DIVERGES: standalone unit compiled vs eager")
        return 3
    print("match: standalone unit does not reproduce; "
          "divergence needs surrounding structure - see the differential and mechanism probes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def emit_repro(out_dir, unit, inputs, model_hint=None):
    """Write repro.npz + repro.py for one unit."""
    os.makedirs(out_dir, exist_ok=True)
    save_inputs(os.path.join(out_dir, "repro.npz"), inputs)
    if unit.repro_import is not None:
        rebuild = unit.repro_import
    elif unit.attr_path is not None:
        rebuild = (
            "def make_fn(args, inputs):\n"
            "    from differential_compile import load_qwen, resolve_path\n"
            "    model, _ = load_qwen(args.model)\n"
            f"    return resolve_path(model, {unit.attr_path!r})\n"
        )
    else:
        rebuild = unit.builder or (
            "def make_fn(args, inputs):\n    raise SystemExit('no builder')\n")
    script = REPRO_TEMPLATE.format(label=unit.label, rebuild=rebuild,
                                   model_hint=model_hint or "", py=sys.executable,
                                   repro=os.path.join(out_dir, "repro.py"),
                                   scripts_dir=SCRIPTS_DIR)
    path = os.path.join(out_dir, "repro.py")
    with open(path, "w") as f:
        f.write(script)
    return out_dir


# --------------------------------------------------------------------------
# Real-path pass: native mlx-lm compile structure, env-isolated subprocesses
# --------------------------------------------------------------------------

REALPATH_CHILD = r'''
import json, os, sys
import numpy as np
import mlx.core as mx

model_path, prompt, steps, out_path = sys.argv[1:5]
inject_step = int(sys.argv[5]) if len(sys.argv) > 5 else -1
steps = int(steps)
mx.random.seed(0)
try:
    from mlx_lm.utils import load
    model, tok = load(model_path)
except ImportError:
    from mlx_lm.utils import load_model
    model, tok = load_model(model_path)
raw = len(sys.argv) > 6 and sys.argv[6] == "1"
if raw:
    ids = tok.encode(prompt)
else:
    try:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True)
    except Exception:
        ids = tok.encode(prompt)

tokens = list(ids)
last_logits = []
generated = []
for s in range(steps):
    logits = model(mx.array([tokens]))
    last = logits[:, -1, :]
    mx.eval(last)
    if s == inject_step:
        last = last + 1.0
    last_logits.append(np.asarray(last.astype(mx.float32)))
    nxt = int(mx.argmax(last))
    generated.append(nxt)
    tokens.append(nxt)

np.savez(out_path, logits=np.stack(last_logits))
json.dump({"tokens": generated,
           "disable": os.environ.get("MLX_DISABLE_COMPILE")},
          open(out_path + ".json", "w"))
'''


def run_realpath(args, disable):
    env = dict(os.environ)
    if disable:
        env["MLX_DISABLE_COMPILE"] = "1"
    else:
        env.pop("MLX_DISABLE_COMPILE", None)
    tag = "eager" if disable else "compiled"
    os.makedirs(args.out_dir, exist_ok=True)
    child = os.path.join(args.out_dir, f"realpath_child_{tag}.py")
    with open(child, "w") as f:
        f.write(REALPATH_CHILD)
    dump = os.path.join(args.out_dir, f"realpath_{tag}.npz")
    cmd = [sys.executable, child, args.model, args.prompt, str(args.steps), dump,
           str(args.inject_step if not disable else -1),
           "1" if args.no_chat_template else "0"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"realpath child ({tag}) failed: exit {r.returncode}")
    d = np.load(dump)
    meta = json.load(open(dump + ".json"))
    return d["logits"], meta


# --------------------------------------------------------------------------
# Self-test: comparator logic, no mlx required
# --------------------------------------------------------------------------

def self_test():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 5)).astype(np.float32)
    assert bitwise_equal(a, a.copy())
    b = a.copy()
    b[2, 3] = np.float32(np.nextafter(b[2, 3], np.float32(np.inf)))
    assert not bitwise_equal(a, b)
    assert first_diff_index(a, b) == 13
    # NaN: identical bit patterns match, different payloads do not
    n1 = a.copy(); n2 = a.copy(); n3 = a.copy()
    n1[0, 0] = np.float32("nan"); n2[0, 0] = np.float32("nan"); n3[0, 0] = np.float32("nan")
    n3[0, 0] = np.float32(np.float64(np.float32("nan")))
    assert bitwise_equal(n1, n2)
    assert not bitwise_equal(n1, a)
    # uint16 travel (bf16 stand-in)
    u = rng.integers(0, 2 ** 16, size=7).astype(np.uint16)
    assert bitwise_equal(u, u.copy())
    v = u.copy(); v[6] = np.uint16(v[6] + 1)
    assert first_diff_index(u, v) == 6
    # detector must fire on an injected one-element perturbation
    cap_a = [{"np": a, "dtype": "float32", "shape": list(a.shape)}]
    assert compare_capture(cap_a, [dict(c, np=c["np"].copy()) for c in cap_a]) is None
    c = a.copy(); c[1, 1] = np.float32(c[1, 1] + 1.0)
    d = compare_capture(cap_a, [{"np": c, "dtype": "float32", "shape": list(a.shape)}])
    assert d is not None and d["kind"] == "value" and d["index"] == 6
    # tolerance mode still reports gross corruption
    d2 = compare_capture(cap_a, [{"np": c, "dtype": "float32", "shape": list(a.shape)}],
                         atol=1e-3)
    assert d2 is not None and d2["kind"] == "value(tolerance)"
    print("self-test: all comparator checks pass")
    return EXIT_MATCH


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="differential compiled-vs-eager harness")
    p.add_argument("--mode", choices=["graph", "model", "realpath", "self-test"],
                   default="graph")
    p.add_argument("--fn", default="chain", help="graph builtin: chain | swiglu")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--model", default=None, help="model path (model/realpath modes)")
    p.add_argument("--prompt", default="What is the capital of France?")
    p.add_argument("--steps", type=int, default=3, help="decode steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inject-unit", type=int, default=None,
                   help="graph: perturb this unit's captured output in the compiled "
                        "pass to prove the detector detects")
    p.add_argument("--inject-out", type=int, default=0)
    p.add_argument("--inject-step", type=int, default=-1,
                   help="realpath: perturb this decode step's logits in the compiled child")
    p.add_argument("--no-chat-template", action="store_true",
                   help="tokenize the raw prompt; avoids the chat-template prefix")
    p.add_argument("--atol", type=float, default=None,
                   help="WARNING: switches to tolerance comparison; default is bitwise exact")
    p.add_argument("--shrink", action="store_true",
                   help="emit standalone reproduction of the first divergence")
    p.add_argument("--shrink-dir", default="repro_shrunk")
    p.add_argument("--out-dir", default="/tmp/differential")
    p.add_argument("--json", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test or args.mode == "self-test":
        return self_test()
    if args.atol is not None:
        print(f"WARNING: --atol {args.atol}: tolerance comparison ENABLED. "
              "Default is bitwise exact.", file=sys.stderr)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "graph":
        import mlx.core as mx

        mx.random.seed(args.seed)
        if args.fn == "chain":
            units, fixed, specs = graph_chain(args.seed, args.size, args.dtype)
        elif args.fn == "swiglu":
            units, fixed, specs = graph_swiglu(args.seed, args.size, args.dtype)
        else:
            raise SystemExit(f"unknown graph fn: {args.fn}")
        mx.disable_compile()
        eager = run_graph(units, fixed, specs, compiled=False)
        inject = None
        if args.inject_unit is not None:
            inject = (args.inject_unit, args.inject_out, 1.0)
        mx.enable_compile()
        compiled = run_graph(units, fixed, specs, compiled=True, inject=inject)
        mx.disable_compile()
        rep = diff_records(eager, compiled, atol=args.atol, step="graph")
        if rep is None:
            print(f"graph '{args.fn}': all units match bitwise "
                  f"(dtype={args.dtype}, size={args.size}, seed={args.seed})")
            _json_out(args, {"status": "match"})
            return EXIT_MATCH
        print(rep.render())
        if args.shrink:
            unit = rep.unit
            if rep.args_arrays:
                div = standalone_diverges(unit, rep.args_arrays, atol=args.atol)
                kept = shrink_inputs(unit, rep.args_arrays, atol=args.atol) if div \
                    else rep.args_arrays
                d = emit_repro(args.shrink_dir, unit, kept)
                print(f"shrink: emitted {d}/repro.py + repro.npz "
                      f"({'standalone diverges' if div else 'context-only; standalone clean'})")
            else:
                print("shrink: no captured input arrays for this divergence")
        _json_out(args, {"status": "diverge", "first": rep.render()})
        return EXIT_DIVERGE

    if args.mode == "model":
        if not args.model:
            raise SystemExit("model mode needs --model")
        model, tok = load_qwen(args.model)
        first, op = model_ladder(model, tok, args)
        if first is None:
            print(f"model: all layer boundaries match bitwise over {args.steps + 1} "
                  "full-recompute forward passes (descent not needed); "
                  "compiled path clean at this granularity on this device")
            _json_out(args, {"status": "match"})
            return EXIT_MATCH
        target = op if op is not None else first
        if args.shrink:
            unit = target.unit
            if unit is not None and target.args_arrays:
                div = standalone_diverges(unit, target.args_arrays, atol=args.atol)
                kept = shrink_inputs(unit, target.args_arrays, atol=args.atol) if div \
                    else target.args_arrays
                d = emit_repro(args.shrink_dir, unit, kept, model_hint=args.model)
                print(f"shrink: emitted {d}/repro.py + repro.npz "
                      f"({'standalone diverges' if div else 'context-only; standalone clean'})")
            else:
                print("shrink: no captured input arrays for this unit")
        _json_out(args, {"status": "diverge", "first": first.render(),
                         "op": op.render() if op is not None else None})
        return EXIT_DIVERGE

    if args.mode == "realpath":
        if not args.model:
            raise SystemExit("realpath mode needs --model")
        lg_e, meta_e = run_realpath(args, disable=True)
        lg_c, meta_c = run_realpath(args, disable=False)
        if lg_e.shape != lg_c.shape:
            print(f"realpath: logit shapes diverge: {lg_e.shape} vs {lg_c.shape}")
            _json_out(args, {"status": "diverge", "at": "shape"})
            return EXIT_DIVERGE
        neq = unsigned_view(lg_e) != unsigned_view(lg_c)
        if neq.any():
            step = int(np.argmax(neq.any(axis=tuple(range(1, neq.ndim)))))
            idx = int(np.argmax(neq[step].reshape(-1)))
            print(f"realpath: logits diverge at decode step {step}, flat index {idx}: "
                  f"eager {float(lg_e[step].reshape(-1)[idx])!r} vs "
                  f"compiled {float(lg_c[step].reshape(-1)[idx])!r}")
            print(f"tokens eager   : {meta_e['tokens']}")
            print(f"tokens compiled: {meta_c['tokens']}")
            _json_out(args, {"status": "diverge", "step": step, "index": idx})
            return EXIT_DIVERGE
        print(f"realpath: per-step logits and tokens match bitwise over {args.steps} steps; "
              "native compiled fragments clean at this commit on this device")
        _json_out(args, {"status": "match"})
        return EXIT_MATCH

    raise SystemExit(f"unhandled mode {args.mode}")


def _json_out(args, payload):
    if args.json:
        print(json.dumps(payload))


if __name__ == "__main__":
    sys.exit(main())
