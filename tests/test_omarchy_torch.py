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


if __name__ == "__main__":
    unittest.main()
