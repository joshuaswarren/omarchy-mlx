"""The wheel-shipped torch backend registers and compiles a tiny module.

Proves the product path: `import mlx.omarchy_torch` registers the
`mlx_vulkan` dynamo backend, and torch.compile with that backend runs a
small module end to end on the GPU. Skipped where mlx or torch is absent
(the backend is opt-in through the omarchy wheel).
"""
import unittest


class TestOmarchyTorchBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mlx  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"mlx/torch unavailable: {exc}")
        try:
            import mlx.omarchy_torch  # registers the backend on import
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(
                f"the installed mlx predates omarchy_torch: {exc}")

    def test_backend_registered(self):
        import torch

        backends = set(torch._dynamo.list_backends())
        self.assertIn("mlx_vulkan", backends)

    def test_tiny_module_compiles_and_runs(self):
        import torch

        class Tiny(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x) * 2 + 1

        compiled = torch.compile(Tiny(), backend="mlx_vulkan")
        x = torch.eye(4)
        out = compiled(x)
        want = torch.relu(torch.eye(4)) * 2 + 1
        self.assertTrue(torch.allclose(out.cpu(), want.cpu()),
                        (out.cpu(), want.cpu()))

    def test_native_layer_norm_backward(self):
        """PT-U2: the wheel backend lowers aten.native_layer_norm_backward —
        the dX/dW/dB triple from the saved mean/rstd — with fp32 parity
        against the aten CPU op."""
        import mlx.core as mx
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        torch.manual_seed(0)
        lowering = ATEN_TABLE["native_layer_norm_backward"]
        for ns_shape, xshape in [((128,), (4, 64, 128)), ((64,), (8, 64)),
                                 ((8, 16), (4, 8, 16))]:
            x = torch.randn(*xshape)
            w = torch.randn(*ns_shape)
            b = torch.randn(*ns_shape)
            out, mean, rstd = torch.native_layer_norm(
                x, list(ns_shape), w, b, 1e-5)
            g = torch.randn_like(out)
            ref = torch.ops.aten.native_layer_norm_backward(
                g, x, list(ns_shape), mean, rstd, w, b, [True, True, True])
            mine = lowering(mx.array(g.numpy()), mx.array(x.numpy()),
                            list(ns_shape), mx.array(mean.numpy()),
                            mx.array(rstd.numpy()), mx.array(w.numpy()),
                            mx.array(b.numpy()), [True, True, True])
            for tag, m, r in zip(("dx", "dw", "db"), mine, ref):
                got = torch.from_numpy(__import__("numpy").array(m))
                self.assertTrue(
                    torch.allclose(got, r, atol=1e-5, rtol=1e-5),
                    (ns_shape, tag, float((got - r).abs().max())))

    def test_log1p_expm1_logaddexp(self):
        """PT-AL-f: the wheel backend lowers aten.log1p, aten.expm1 and
        aten.logaddexp with fp32/bf16 parity against the aten CPU ops,
        including near-zero inputs where the naive log(1+x)/exp(x)-1
        forms lose precision."""
        import mlx.core as mx
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        torch.manual_seed(0)
        # near-zero inputs where log(1+x) and exp(x)-1 are catastrophic
        x = torch.tensor([0.0, 1e-9, -1e-9, 1e-7, -1e-7, 1e-4, -1e-4,
                          1e-2, -1e-2, -0.5, 0.5, 1.0, -0.9999, 2.0])
        y = torch.tensor([0.0, 1e-9, -1e-9, -20.0, 20.0, 1e-4, -1.0,
                          0.5, -0.5, 3.0, -3.0, 10.0, -10.0, 0.001])
        cases = [
            ("log1p", torch.ops.aten.log1p.default, (x,)),
            ("expm1", torch.ops.aten.expm1.default, (x,)),
            ("logaddexp", torch.ops.aten.logaddexp.default, (x, y)),
        ]
        np = __import__("numpy")
        for name, aten_fn, args in cases:
            lowering = ATEN_TABLE[name]
            for dtype in (torch.float32, torch.bfloat16):
                targs = [a.to(dtype) for a in args]
                ref = aten_fn(*targs)
                margs = [
                    mx.array(a.float().numpy()).astype(
                        mx.float32 if dtype == torch.float32 else mx.bfloat16)
                    for a in targs
                ]
                got = torch.from_numpy(
                    np.array(lowering(*margs).astype(mx.float32)))
                tol = dict(atol=1e-6, rtol=1e-5) if dtype == torch.float32 \
                    else dict(atol=1e-2, rtol=1e-2)
                self.assertTrue(
                    torch.allclose(got, ref.float(), **tol),
                    (name, dtype, float((got - ref.float()).abs().max())))


    def test_searchsorted_bucketize_parity(self):
        """PT-IDX-a: the wheel backend lowers aten.searchsorted (right,
        out_int32) and aten.bucketize — int64/fp32 parity vs torch CPU."""
        import numpy as np
        import mlx.core as mx
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        ss = ATEN_TABLE["searchsorted"]
        bz = ATEN_TABLE["bucketize"]
        cases = [
            (torch.tensor([1, 3, 3, 5, 7, 9]),
             torch.tensor([0, 3, 4, 9, 10])),
            (torch.tensor([1.0, 3.0, 3.0, 5.0, 7.0, 9.0]),
             torch.tensor([0.0, 3.0, 3.5, 9.0, 10.0])),
        ]
        for seq, vals in cases:
            mseq = mx.array(seq.numpy())
            mvals = mx.array(vals.numpy())
            for right in (False, True):
                ref = torch.searchsorted(seq, vals, right=right)
                got = torch.from_numpy(np.array(ss(mseq, mvals, right=right)))
                self.assertEqual(got.dtype, torch.int64)
                self.assertTrue(torch.equal(got, ref), (seq, vals, right, got, ref))
            ref_b = torch.bucketize(vals, seq)
            got_b = torch.from_numpy(np.array(bz(mvals, mseq)))
            self.assertEqual(got_b.dtype, torch.int64)
            self.assertTrue(torch.equal(got_b, ref_b), (seq, vals, got_b, ref_b))
            ref32 = torch.searchsorted(seq, vals, out_int32=True)
            got32 = torch.from_numpy(np.array(ss(mseq, mvals, out_int32=True)))
            self.assertEqual(got32.dtype, torch.int32)
            self.assertTrue(torch.equal(got32, ref32), (seq, vals, got32, ref32))



    def test_piecewise_activations(self):
        """PT-AL-b: the piecewise activation family lowers in the wheel
        backend — fp32 and bf16 parity against the torch functions."""
        import mlx.core as mx
        import numpy as np
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        x = torch.tensor([[-4.0, -2.9, -0.49, 0.0, 0.51, 3.1, 7.0],
                          [-3.2, -0.5, 0.5, 1.0, -1.0, 2.9, -7.0]])
        F = torch.nn.functional
        cases = [
            ("hardswish", (x,), F.hardswish(x)),
            ("hardsigmoid", (x,), F.hardsigmoid(x)),
            ("hardshrink", (x, 0.5), F.hardshrink(x, 0.5)),
            ("softshrink", (x, 0.5), F.softshrink(x, 0.5)),
            ("tanhshrink", (x,), F.tanhshrink(x)),
            ("softsign", (x,), F.softsign(x)),
        ]
        for name, ins, ref in cases:
            lowering = ATEN_TABLE[name]
            for dtype, tol in ((torch.float32, 1e-5),
                               (torch.bfloat16, 2e-2)):
                mins = tuple(
                    (mx.array(t.float().numpy()).astype(mx.bfloat16)
                     if dtype == torch.bfloat16 else mx.array(t.numpy()))
                    if torch.is_tensor(t) else t for t in ins)
                got = torch.from_numpy(
                    np.array(lowering(*mins).astype(mx.float32)))
                self.assertTrue(
                    torch.allclose(got, ref.float(), atol=tol, rtol=tol),
                    (name, dtype, float((got - ref.float()).abs().max())))

    def test_im2col_col2im_parity(self):
        """PT-IM-a: the explicit im2col path — fp32 parity vs the aten CPU ops
        across stride/padding/dilation, then col2im round-trip."""
        import mlx.core as mx
        import torch
        import torch.nn.functional as F

        from mlx.omarchy_torch import _col2im, _im2col

        cases = [
            dict(n=2, c=3, h=7, w=8, kh=3, kw=3, stride=1, padding=0, dilation=1),
            dict(n=1, c=4, h=9, w=9, kh=2, kw=2, stride=2, padding=1, dilation=1),
            dict(n=2, c=2, h=10, w=8, kh=3, kw=2, stride=2, padding=2, dilation=2),
        ]
        for case in cases:
            torch.manual_seed(7)
            x = torch.randn(case["n"], case["c"], case["h"], case["w"], dtype=torch.float32)
            want = F.unfold(x, (case["kh"], case["kw"]), dilation=case["dilation"],
                            padding=case["padding"], stride=case["stride"])
            got = _im2col(mx.array(x.numpy()), (case["kh"], case["kw"]),
                          dilation=case["dilation"], padding=case["padding"],
                          stride=case["stride"])
            self.assertTrue(mx.allclose(got, mx.array(want.numpy()), atol=0, rtol=0),
                            (case, got.shape))
            back = _col2im(got, (case["h"], case["w"]), (case["kh"], case["kw"]),
                           dilation=case["dilation"], padding=case["padding"],
                           stride=case["stride"])
            want_back = F.fold(want, (case["h"], case["w"]),
                               (case["kh"], case["kw"]), dilation=case["dilation"],
                               padding=case["padding"], stride=case["stride"])
            self.assertTrue(mx.allclose(back, mx.array(want_back.numpy()), atol=0, rtol=0),
                            (case, back.shape))





    def test_elu_family(self):
        """PT-AL-a: elu/selu/celu lower in the wheel backend — fp32 and
        bf16 parity against the torch functions, including the
        scale/input_scale parameters and the in-place aten aliases."""
        import mlx.core as mx
        import numpy as np
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        x = torch.tensor([[-8.0, -2.5, -0.3, 0.0], [0.25, 1.5, 6.0, -0.001]])
        F = torch.nn.functional
        cases = [
            ("elu", (x,), F.elu(x)),
            ("elu", (x, 2.0), F.elu(x, alpha=2.0)),
            ("elu", (x, 1.5, 1.2, 0.8),
             torch.ops.aten.elu.default(x, 1.5, 1.2, 0.8)),
            ("selu", (x,), F.selu(x)),
            ("celu", (x,), F.celu(x)),
            ("celu", (x, 2.5), F.celu(x, alpha=2.5)),
            ("elu_", (x, 1.7), F.elu(x, alpha=1.7)),
            ("selu_", (x,), F.selu(x)),
            ("celu_", (x, 2.2), F.celu(x, alpha=2.2)),
        ]
        for name, ins, ref in cases:
            lowering = ATEN_TABLE[name]
            for dtype, tol in ((torch.float32, 1e-5),
                               (torch.bfloat16, 2e-2)):
                mins = tuple(
                    (mx.array(t.float().numpy()).astype(mx.bfloat16)
                     if dtype == torch.bfloat16 else mx.array(t.numpy()))
                    if torch.is_tensor(t) else t for t in ins)
                got = torch.from_numpy(
                    np.array(lowering(*mins).astype(mx.float32)))
                self.assertTrue(
                    torch.allclose(got, ref.float(), atol=tol, rtol=tol),
                    (name, ins[1:], dtype,
                     float((got - ref.float()).abs().max())))


if __name__ == "__main__":
    unittest.main()
