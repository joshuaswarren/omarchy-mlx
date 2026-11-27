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
            want_back = F.fold(mx.array(want.numpy()), (case["h"], case["w"]),
                               (case["kh"], case["kw"]), dilation=case["dilation"],
                               padding=case["padding"], stride=case["stride"])
            self.assertTrue(mx.allclose(back, mx.array(want_back.numpy()), atol=0, rtol=0),
                            (case, back.shape))


if __name__ == "__main__":
    unittest.main()
