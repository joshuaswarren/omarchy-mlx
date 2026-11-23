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


if __name__ == "__main__":
    unittest.main()
