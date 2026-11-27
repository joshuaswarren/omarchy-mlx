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

    def test_elu_selu_celu_parity(self):
        """PT-AL-a: the wheel backend lowers aten.elu, aten.selu and
        aten.celu with fp32/bf16 parity against the aten CPU ops,
        including the alpha/scale/input_scale parameters and the
        negative region where exp(x)-1 loses precision naively."""
        import mlx.core as mx
        import torch

        from mlx.omarchy_torch import ATEN_TABLE

        torch.manual_seed(0)
        x = torch.tensor([0.0, -1e-9, 1e-9, -1e-4, 1e-4, -0.5, 0.5,
                          -1.0, 1.0, -2.0, 2.0, -10.0, 10.0, 0.25])
        cases = [
            ("elu", torch.ops.aten.elu.default, (x, 0.5, 1.5, 1.25)),
            ("selu", torch.ops.aten.selu.default, (x,)),
            ("celu", torch.ops.aten.celu.default, (x, 0.7)),
        ]
        np = __import__("numpy")
        for name, aten_fn, args in cases:
            lowering = ATEN_TABLE[name]
            for dtype in (torch.float32, torch.bfloat16):
                targs = [args[0].to(dtype), *args[1:]]
                ref = aten_fn(*targs)
                margs = [
                    mx.array(args[0].float().numpy()).astype(
                        mx.float32 if dtype == torch.float32 else mx.bfloat16)
                ]
                got = torch.from_numpy(
                    np.array(lowering(*margs, *args[1:]).astype(mx.float32)))
                tol = dict(atol=1e-6, rtol=1e-5) if dtype == torch.float32 \
                    else dict(atol=4e-2, rtol=4e-2)
                torch.testing.assert_close(
                    got, ref.float(), **tol,
                    msg=lambda m, n=name, d=dtype: f"{n} {d}: {m}")

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


if __name__ == "__main__":
    unittest.main()
