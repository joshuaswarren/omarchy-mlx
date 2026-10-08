"""Independent arbiter for the fused flux2 double norm+rope kernel: runs the
custom kernel and a plain mx-composed pipeline over the SAME inputs and prints
both outputs plus elementwise maxdiffs. Cuts the three-way ambiguity between
the kernel, the numpy emulation, and a reference bug.

GPU host only (imports mlx):
    python3 -m tools.kernel_recheck.arbiter_flux [double|single]
"""
import sys

import numpy as np

from . import defs
from .run import build_inputs, source_text


def gather_qkv(inp):
    """Per (s, h) q/k/v planes gathered exactly as the kernel indexes them."""
    d = defs.FLUX
    dim, hd = d["dim"], d["hd"]
    s_img, s_txt = d["img"], d["txt"]
    s_tot = s_img + s_txt
    txt = inp["txt_qkv"][0].reshape(s_txt, 3 * dim)
    img = inp["img_qkv"][0].reshape(s_img, 3 * dim)
    q = np.zeros((s_tot, heads := d["heads"], hd), dtype=np.float32)
    k = np.zeros_like(q)
    v = np.zeros_like(q)
    for s in range(s_tot):
        src = txt[s] if s < s_txt else img[s - s_txt]
        for h in range(d["heads"]):
            q[s, h] = src[h * hd:(h + 1) * hd]
            k[s, h] = src[dim + h * hd:dim + (h + 1) * hd]
            v[s, h] = src[2 * dim + h * hd:2 * dim + (h + 1) * hd]
    return q, k, v


def mx_double_ref(inp):
    """Composed-mx reference: bf16 rms-norm with per-row weights (added norms
    on txt rows), then interleaved rope with the cos/sin tables."""
    import mlx.core as mx
    d = defs.FLUX
    eps = d["eps"]
    s_img, s_txt = d["img"], d["txt"]
    q, k, v = gather_qkv(inp)
    to_bf16 = lambda a: mx.array(a).astype(mx.bfloat16)
    q, k, v = to_bf16(q), to_bf16(k), to_bf16(v)
    nq = to_bf16(np.concatenate([np.repeat(inp["norm_added_q"][None], s_txt, 0),
                                 np.repeat(inp["norm_q"][None], s_img, 0)]))
    nk = to_bf16(np.concatenate([np.repeat(inp["norm_added_k"][None], s_txt, 0),
                                 np.repeat(inp["norm_k"][None], s_img, 0)]))
    def rms(x, w):
        f = x.astype(mx.float32)
        inv = mx.rsqrt(mx.mean(f * f, axis=-1, keepdims=True) + eps)
        return (x.astype(mx.float32) * inv).astype(mx.bfloat16) * w[:, None, :]
    q, k = rms(q, nq), rms(k, nk)
    cos = to_bf16(inp["cos_vals"])[:, None, :].astype(mx.float32)
    sin = to_bf16(inp["sin_vals"])[:, None, :].astype(mx.float32)
    def rope(x):
        f = x.astype(mx.float32)
        a, b = f[..., 0::2], f[..., 1::2]
        inter = mx.stack([a * cos - b * sin, b * cos + a * sin], axis=-1)
        return inter.reshape(f.shape).astype(mx.bfloat16)
    return [o.reshape(1, o.shape[0], o.shape[1], o.shape[2])
            for o in (rope(q), rope(k), v)]


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "double"
    import mlx.core as mx
    spec = defs.get_spec("fused_double_norm_rope" if which == "double"
                         else "fused_single_norm_rope")
    inputs = build_inputs(spec)
    kernel = mx.fast.metal_kernel(
        name=spec["name"] + "_arbiter",
        input_names=list(spec["inputs"]),
        output_names=list(spec["outputs"]),
        source=source_text(spec),
    )
    arrays = [mx.array(np.ascontiguousarray(inputs[n], dtype=np.float32)).astype(mx.bfloat16)
              for n in spec["inputs"]]
    outs = kernel(inputs=arrays, template=[("T", mx.bfloat16)], grid=spec["grid"],
                  threadgroup=spec["threadgroup"],
                  output_shapes=[tuple(s) for s in spec["out_shapes"]],
                  output_dtypes=[mx.bfloat16] * 3, stream=mx.gpu)
    for o in outs:
        mx.eval(o)
    got = [np.array(o.astype(mx.float32)).astype(np.float64) for o in outs]
    emu = spec["ref"](inputs)
    comp = mx_double_ref(inputs) if which == "double" else None
    for i, g in enumerate(got):
        line = (f"out[{i}] kernel-vs-emulation maxdiff "
                f"{np.abs(g - emu[i]).max():.4g}")
        if comp is not None:
            c = np.array(comp[i].astype(mx.float32)).astype(np.float64)
            line += (f" | kernel-vs-mx-composed maxdiff "
                     f"{np.abs(g - c).max():.4g}")
        print(line)
    if comp is not None:
        print("kernel[0,:2,0,:4]   ", got[0][0, :2, 0, :4].ravel())
        c = np.array(comp[0].astype(mx.float32)).astype(np.float64)
        print("mxcomp[0,:2,0,:4]  ", c[0, :2, 0, :4].ravel())


if __name__ == "__main__":
    main()
