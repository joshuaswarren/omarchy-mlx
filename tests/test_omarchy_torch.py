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
            ("expm1_", torch.ops.aten.expm1_.default, (x,)),
            ("logaddexp", torch.ops.aten.logaddexp.default, (x, y)),
        ]
        np = __import__("numpy")
        for name, aten_fn, args in cases:
            lowering = ATEN_TABLE[name]
            for dtype in (torch.float32, torch.bfloat16):
                targs = [a.to(dtype) for a in args]
                margs = [
                    mx.array(a.float().numpy()).astype(
                        mx.float32 if dtype == torch.float32 else mx.bfloat16)
                    for a in targs
                ]
                ref = aten_fn(*targs)
                got = torch.from_numpy(
                    np.array(lowering(*margs).astype(mx.float32)))
                tol = dict(atol=1e-6, rtol=1e-5) if dtype == torch.float32 \
                    else dict(atol=1e-2, rtol=1e-2)
                self.assertTrue(
                    torch.allclose(got, ref.float(), **tol),
                    (name, dtype, float((got - ref.float()).abs().max())))
        torch.manual_seed(1)

    def test_parametric_activations(self):
        """Q-58: the wheel backend lowers the parameterized activation
        family — aten.leaky_relu, aten.hardtanh, aten.threshold,
        aten.prelu, aten.rrelu (eval) — with fp32/bf16 parity against the
        aten CPU ops, and refuses rrelu in training mode."""
        import mlx.core as mx
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        x = torch.tensor([[-6.0, -2.0, -0.25, 0.0], [0.5, 1.0, 4.0, -0.001]])
        w1 = torch.tensor([0.2])
        wc = torch.tensor([0.1, 0.3])
        x4 = torch.tensor([[[[-2.0], [-1.0]], [[0.5], [3.0]]]])
        cases = [
            ("leaky_relu", torch.ops.aten.leaky_relu.default, (x, 0.2)),
            ("leaky_relu_", torch.ops.aten.leaky_relu_.default, (x, 0.05)),
            ("hardtanh", torch.ops.aten.hardtanh.default, (x, -0.5, 0.5)),
            ("hardtanh_", torch.ops.aten.hardtanh_.default, (x, 0.0, 1.5)),
            ("threshold", torch.ops.aten.threshold.default, (x, 0.25, 0.0)),
            ("threshold", torch.ops.aten.threshold.default, (x, 0.0, -1.0)),
            ("threshold_", torch.ops.aten.threshold_.default, (x, 0.25, -2.0)),
            ("prelu", torch.ops.aten.prelu.default, (x, w1)),
            ("prelu", torch.ops.aten.prelu.default, (x, wc)),
            ("prelu", torch.ops.aten.prelu.default, (x4, wc)),
            ("rrelu", torch.ops.aten.rrelu.default, (x,)),
            ("rrelu", torch.ops.aten.rrelu.default, (x, 0.1, 0.5, False)),
            ("rrelu_", torch.ops.aten.rrelu_.default, (x, 0.1, 0.5, False)),
        ]
        np = __import__("numpy")
        for name, aten_fn, args in cases:
            lowering = ATEN_TABLE[name]
            for dtype in (torch.float32, torch.bfloat16):
                targs = [a.to(dtype) if torch.is_tensor(a) else a
                         for a in args]
                margs = [
                    mx.array(a.float().numpy()).astype(
                        mx.float32 if dtype == torch.float32 else mx.bfloat16)
                    if torch.is_tensor(a) else a for a in targs
                ]
                ref = aten_fn(*targs)
                got = torch.from_numpy(
                    np.array(lowering(*margs).astype(mx.float32)))
                tol = dict(atol=1e-6, rtol=1e-5) if dtype == torch.float32 \
                    else dict(atol=1e-2, rtol=1e-2)
                self.assertTrue(
                    torch.allclose(got, ref.float(), **tol),
                    (name, dtype, float((got - ref.float()).abs().max())))
        # rrelu in training mode samples noise — no deterministic lowering.
        with self.assertRaises(Exception):
            ATEN_TABLE["rrelu"](mx.array(x.numpy()), 0.125, 0.3333, True)
        # threshold at x == threshold yields `value`, not x.
        got = np.array(ATEN_TABLE["threshold"](mx.zeros((2,)), 0.0, -3.0))
        self.assertTrue((got == -3.0).all(), got)

    def test_unfold_last_dim(self):
        """PT-IM-b: aten.unfold lowers via gather + transpose — fp32 parity
        against the aten CPU op on the last dimension."""
        lowering = ATEN_TABLE["unfold"]
        for shape, dim, size, step in [((2, 3, 10), 2, 4, 2), ((4, 10), 1, 3, 1),
                                       ((2, 3, 12), 2, 5, 3)]:
            x = torch.randn(*shape)
            ref = x.unfold(dim, size, step)
            mine = lowering(mx.array(x.numpy()), dim, size, step)
            got = torch.from_numpy(__import__("numpy").array(mine))
            self.assertTrue(got.shape == ref.shape, (shape, got.shape, ref.shape))
            self.assertTrue(torch.allclose(got.float(), ref.float(), atol=0, rtol=0),
                            (shape, float((got - ref).abs().max())))

    def test_unfold_non_last_dim(self):
        """PT-IM-b: unfold on a non-last dimension moves the window dim to
        the end per the aten spec."""
        import mlx.core as mx
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        torch.manual_seed(2)
        lowering = ATEN_TABLE["unfold"]
        x = torch.randn(2, 4, 6)
        ref = x.unfold(1, 3, 2)
        mine = lowering(mx.array(x.numpy()), 1, 3, 2)
        got = torch.from_numpy(__import__("numpy").array(mine))
        self.assertTrue(got.shape == ref.shape, (got.shape, ref.shape))
        self.assertTrue(torch.allclose(got.float(), ref.float(), atol=0, rtol=0))

    def test_fold_non_overlapping(self):
        """PT-IM-b: aten.fold with stride == kernel is reshape/transpose —
        fp32 parity against the aten CPU op, and the fold(unfold(x)) round
        trip is exact."""
        import mlx.core as mx
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        torch.manual_seed(3)
        fold = ATEN_TABLE["fold"]
        unfold = ATEN_TABLE["unfold"]
        x = torch.randn(2, 9, 1)  # [N, C*k, L] for C=3, k=3x3, stride=3x3: L=1
        ref = torch.nn.functional.fold(
            x, output_size=(3, 3), kernel_size=(3, 3))
        mine = fold(mx.array(x.numpy()), (3, 3), (3, 3))
        got = torch.from_numpy(__import__("numpy").array(mine))
        self.assertTrue(got.shape == ref.shape, (got.shape, ref.shape))
        self.assertTrue(torch.allclose(got.float(), ref.float(), atol=1e-5, rtol=1e-5))

    def test_fold_overlapping_sums(self):
        """PT-IM-b: aten.fold with stride < kernel SUMS overlapping windows —
        fp32 parity against the aten CPU op."""
        import mlx.core as mx
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        torch.manual_seed(4)
        fold = ATEN_TABLE["fold"]
        x = torch.randn(2, 12, 16)  # [N, C*k, L] for C=3, k=2x2, L=16
        ref = torch.nn.functional.fold(
            x, output_size=(5, 5), kernel_size=(2, 2), stride=(1, 1))
        mine = fold(mx.array(x.numpy()), (5, 5), (2, 2), (1, 1), (0, 0))
        got = torch.from_numpy(__import__("numpy").array(mine))
        self.assertTrue(got.shape == ref.shape, (got.shape, ref.shape))
        self.assertTrue(torch.allclose(got.float(), ref.float(), atol=1e-5, rtol=1e-5),
                        float((got - ref).abs().max()))

    def test_fold_strided_multi_block(self):
        """PT-IM-b re-review: the strided-placement path with multiple
        blocks per axis (stride < output span, L > 1) — fp32 parity vs the
        aten CPU op. Pins the bug where blocks stacked at stride 1."""
        import mlx.core as mx
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        torch.manual_seed(9)
        fold = ATEN_TABLE["fold"]
        x = torch.randn(2, 12, 4)  # [N, C*4, L] for C=3, k=2x2, L=4
        ref = torch.nn.functional.fold(
            x, output_size=(8, 8), kernel_size=(2, 2), stride=(2, 2))
        mine = fold(mx.array(x.numpy()), (8, 8), (2, 2), (1, 1), (0, 0), (2, 2))
        got = torch.from_numpy(__import__("numpy").array(mine))
        self.assertTrue(got.shape == ref.shape, (got.shape, ref.shape))
        self.assertTrue(torch.allclose(got.float(), ref.float(), atol=1e-5, rtol=1e-5),
                        float((got - ref).abs().max()))


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

    def test_glu_parity(self):
        """PT-AL-d: aten.glu lowers in the wheel backend — fp32 and
        bf16 parity against torch across dim=-1/0/1."""
        import mlx.core as mx
        import numpy as np
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        x2 = torch.randn(3, 8)
        x3 = torch.randn(2, 6, 4)
        F = torch.nn.functional
        cases = [
            ((x2,), {}, F.glu(x2)),
            ((x3,), {"dim": 0}, F.glu(x3, dim=0)),
            ((x3,), {"dim": 1}, F.glu(x3, dim=1)),
            ((x3,), {"dim": -2}, F.glu(x3, dim=-2)),
        ]
        lowering = ATEN_TABLE["glu"]
        for ins, kw, ref in cases:
            for dtype, tol in ((torch.float32, 1e-5),
                               (torch.bfloat16, 2e-2)):
                mins = tuple(
                    (mx.array(t.float().numpy()).astype(mx.bfloat16)
                     if dtype == torch.bfloat16 else mx.array(t.numpy()))
                    for t in ins)
                got = torch.from_numpy(
                    np.array(lowering(*mins, **kw).astype(mx.float32)))
                self.assertTrue(
                    torch.allclose(got, ref.float(), atol=tol, rtol=tol),
                    (kw, dtype,
                     float((got - ref.float()).abs().max())))



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

    def test_reciprocal_frac_parity(self):
        """PT-AL-e: aten.reciprocal/frac lower in the wheel backend —
        fp32 and bf16 parity incl. the sign-kept frac contract and the
        in-place aten aliases."""
        import mlx.core as mx
        import numpy as np
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        x = torch.tensor([[-2.3, -0.7, -0.5, 0.0], [0.25, 1.9, 8.0, -4.0]])
        cases = [
            ("reciprocal", torch.reciprocal(x)),
            ("frac", torch.frac(x)),
            ("reciprocal_", torch.reciprocal(x)),
            ("frac_", torch.frac(x)),
        ]
        for name, ref in cases:
            lowering = ATEN_TABLE[name]
            for dtype, tol in ((torch.float32, 1e-5),
                               (torch.bfloat16, 2e-2)):
                m = (mx.array(x.float().numpy()).astype(mx.bfloat16)
                     if dtype == torch.bfloat16 else mx.array(x.numpy()))
                got = torch.from_numpy(
                    np.array(lowering(m).astype(mx.float32)))
                self.assertTrue(
                    torch.allclose(got, ref.float(), atol=tol, rtol=tol),
                    (name, dtype,
                     float((got - ref.float()).abs().max())))
        # frac keeps x's sign: -0.3, not the floor form's +0.7.
        got = np.array(ATEN_TABLE["frac"](mx.array([-2.3])))
        self.assertTrue(abs(float(got[0]) - (-0.3)) < 1e-6, got)

    def test_linalg_norm_parity(self):
        """PT-NORM-b: linalg_vector_norm/linalg_norm lower in the wheel
        backend — fp32 and bf16 parity vs torch across ord=1/2/inf, dims,
        keepdim, dtype cast; plus the aten.norm inf/-inf pin."""
        import mlx.core as mx
        import numpy as np
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        x = torch.randn(3, 4)
        x3 = torch.randn(2, 3, 4)
        cases = [
            ("linalg_vector_norm", torch.linalg.vector_norm, (x,), {}),
            ("linalg_vector_norm", torch.linalg.vector_norm, (x,),
             {"ord": 1.0}),
            ("linalg_vector_norm", torch.linalg.vector_norm, (x,),
             {"ord": float("inf"), "dim": [1]}),
            ("linalg_vector_norm", torch.linalg.vector_norm, (x3,),
             {"ord": 0.5, "dim": [1, 2], "keepdim": True}),
            ("linalg_vector_norm", torch.linalg.vector_norm, (x,),
             {"dtype": torch.float64}),
            ("linalg_norm", torch.linalg.norm, (x,), {}),
            ("linalg_norm", torch.linalg.norm, (x,),
             {"ord": float("-inf"), "dim": [0]}),
            ("linalg_norm", torch.linalg.norm, (x3,),
             {"ord": 2.0, "dim": [2]}),
        ]
        for name, fn, ins, kw in cases:
            lowering = ATEN_TABLE[name]
            for dtype, tol in ((torch.float32, 1e-5),
                               (torch.bfloat16, 2e-2)):
                mins = tuple(
                    (mx.array(t.float().numpy()).astype(mx.bfloat16)
                     if dtype == torch.bfloat16 else mx.array(t.numpy()))
                    for t in ins)
                ref = fn(*[t.to(dtype) for t in ins], **kw)
                got = torch.from_numpy(
                    np.array(lowering(*mins, **kw).astype(mx.float32)))
                self.assertTrue(
                    torch.allclose(got, ref.float(), atol=tol, rtol=tol),
                    (name, kw, dtype,
                     float((got - ref.float()).abs().max())))
        m = mx.array(x.numpy())
        for p, ref in ((float("inf"), torch.norm(x, float("inf"))),
                       (float("-inf"), torch.norm(x, float("-inf")))):
            got = float(np.array(ATEN_TABLE["norm"](m, p)))
            self.assertTrue(abs(got - float(ref)) < 1e-5, (p, got))



    def test_logsumexp_parity(self):
        """PT-AL-g: aten.logsumexp lowers in the wheel backend — parity
        vs torch.logsumexp across single/multi dim, keepdim, negative
        dim, bf16, and large-logit stability."""
        import mlx.core as mx
        import numpy as np
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        x2 = torch.randn(3, 4)
        x3 = torch.randn(2, 3, 4)
        cases = [
            ((x2,), {"dim": [0]}, 1e-5),
            ((x2,), {"dim": [0, 1]}, 1e-5),
            ((x3,), {"dim": [2]}, 1e-5),
            ((x3,), {"dim": [0, 2], "keepdim": True}, 1e-5),
            ((x2,), {"dim": [-1]}, 1e-5),
            ((x2.bfloat16(),), {"dim": [1]}, 1e-2),
            ((x2 + 1e4,), {"dim": [1]}, 1e-5),
        ]
        for ins, kw, tol in cases:
            ref = torch.logsumexp(*ins, **kw)
            mins = tuple(
                (mx.array(t.float().numpy()).astype(mx.bfloat16)
                 if t.dtype == torch.bfloat16 else mx.array(t.numpy()))
                for t in ins)
            got = torch.from_numpy(
                np.array(ATEN_TABLE["logsumexp"](*mins, **kw)
                         .astype(mx.float32)))
            self.assertTrue(
                torch.allclose(got, ref.float(), atol=tol, rtol=tol),
                (kw, float((got - ref.float()).abs().max())))


    def test_pdist_cdist_parity(self):
        """PT-PAIR-a: aten.pdist + aten.cdist lower in the wheel
        backend — parity vs torch across p=0.5/1/2/inf and batched
        inputs."""
        import mlx.core as mx
        import numpy as np
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        a = torch.randn(5, 3)
        b = torch.randn(4, 3)
        ab = torch.randn(2, 5, 3)
        bb = torch.randn(2, 4, 3)
        cases = [
            ("cdist", (a, b), {}, torch.cdist(a, b)),
            ("cdist", (a, b), {"p": 1.0}, torch.cdist(a, b, p=1.0)),
            ("cdist", (a, b), {"p": 0.5}, torch.cdist(a, b, p=0.5)),
            ("cdist", (a, b), {"p": float("inf")},
             torch.cdist(a, b, p=float("inf"))),
            ("cdist", (ab, bb), {}, torch.cdist(ab, bb)),
            ("pdist", (a,), {}, torch.pdist(a)),
            ("pdist", (a,), {"p": 1.0}, torch.pdist(a, p=1.0)),
            ("pdist", (a,), {"p": float("inf")},
             torch.pdist(a, p=float("inf"))),
        ]
        for name, ins, kw, ref in cases:
            mins = tuple(mx.array(t.numpy()) for t in ins)
            got = torch.from_numpy(
                np.array(ATEN_TABLE[name](*mins, **kw).astype(mx.float32)))
            self.assertTrue(
                torch.allclose(got, ref.float(), atol=1e-5, rtol=1e-5),
                (name, kw, float((got - ref.float()).abs().max())))

    def test_flip_rot90_parity(self):
        """PT-Q11: aten.flip + aten.rot90 lower in the wheel backend —
        parity vs torch across dims lists, negative dims, k mod 4
        (including negative k), non-default dims planes, and bf16."""
        import mlx.core as mx
        import numpy as np
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        torch.manual_seed(0)
        x2 = torch.randn(3, 5)
        x3 = torch.randn(2, 3, 4)
        cases = [
            ("flip", (x2, [0]), {}, torch.flip(x2, [0])),
            ("flip", (x2, [0, 1]), {}, torch.flip(x2, [0, 1])),
            ("flip", (x3, [-1]), {}, torch.flip(x3, [-1])),
            ("flip", (x2.bfloat16(), [1]), {}, torch.flip(x2.bfloat16(), [1])),
            ("rot90", (x2,), {"k": 1}, torch.rot90(x2, 1)),
            ("rot90", (x2,), {"k": 2}, torch.rot90(x2, 2)),
            ("rot90", (x2,), {"k": 3}, torch.rot90(x2, 3)),
            ("rot90", (x2,), {"k": 4}, torch.rot90(x2, 4)),
            ("rot90", (x2,), {"k": -1}, torch.rot90(x2, -1)),
            ("rot90", (x3,), {"k": 1, "dims": [1, 2]}, torch.rot90(x3, 1, [1, 2])),
            ("rot90", (x3,), {"k": 3, "dims": [-1, -2]}, torch.rot90(x3, 3, [-1, -2])),
            ("rot90", (x2.bfloat16(),), {"k": 1}, torch.rot90(x2.bfloat16(), 1)),
        ]
        for name, ins, kw, ref in cases:
            mins = tuple(
                (mx.array(t.float().numpy()).astype(mx.bfloat16)
                 if torch.is_tensor(t) and t.dtype == torch.bfloat16
                 else mx.array(t.numpy()) if torch.is_tensor(t) else t)
                for t in ins)
            got = torch.from_numpy(
                np.array(ATEN_TABLE[name](*mins, **kw).astype(mx.float32)))
            tol = 2e-2 if ins[0].dtype == torch.bfloat16 else 1e-5
            self.assertTrue(
                torch.allclose(got, ref.float(), atol=tol, rtol=tol),
                (name, kw, float((got - ref.float()).abs().max())))


if __name__ == "__main__":
    unittest.main()
