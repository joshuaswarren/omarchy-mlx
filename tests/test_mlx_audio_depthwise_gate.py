"""Pinned mlx-audio 0.5.6 guard backport, from upstream 4e429cfaf26b.

The fixture is verbatim from wheel sha256
7cf7b49135f6f681988a9e0b9d2fbfdd2e9be783b0a3204d0c4028e0cf2d8b4c.
Run stdlib checks normally. Add --runtime for real MLX numeric/routing checks;
--unpatched makes that regression fail against the original wheel source.
The runtime check requires an explicitly selected llvmpipe ICD, not hardware.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/mlx_audio_056/depthwise_conv1d_kernel.py"
PATCH = ROOT / "patches/mlx-audio-mossformer-depthwise-metal-gate.patch"
MODULE = Path("mlx_audio/sts/models/mossformer2_se/depthwise_conv1d_kernel.py")
RUNTIME = "--runtime" in sys.argv
UNPATCHED = "--unpatched" in sys.argv
for option in ("--runtime", "--unpatched"):
    if option in sys.argv:
        sys.argv.remove(option)


def materialize(root, patched=True):
    target = root / MODULE
    target.parent.mkdir(parents=True)
    target.write_bytes(FIXTURE.read_bytes())
    if patched:
        subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(PATCH)],
                       cwd=root, check=True, capture_output=True, text=True)
    return target


class AudioDepthwiseGate(unittest.TestCase):
    def test_pinned_source_and_exact_backport(self):
        original = FIXTURE.read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(),
                         "7406560a354ae0a3fe322339608e1afdbaff02c92b22997a665c9a87455e712f")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = materialize(root)
            self.assertEqual(target.read_bytes().replace(
                b"        and mx.metal.is_available()\n", b""), original)
            subprocess.run(["patch", "--batch", "--fuzz=0", "--dry-run", "-R", "-p1", "-i", str(PATCH)],
                           cwd=root, check=True, capture_output=True, text=True)

    @unittest.skipUnless(RUNTIME, "pass --runtime to use real MLX on llvmpipe")
    def test_nonmetal_route_and_scalar_reference(self):
        self.assertEqual(os.environ.get("MLX_OMARCHY_ALLOW_NON_APPLE"), "1")
        self.assertIn("lvp_icd", os.environ.get("VK_DRIVER_FILES", ""))
        import mlx.core as mx
        import numpy as np
        self.assertFalse(mx.metal.is_available())
        with tempfile.TemporaryDirectory() as tmp:
            target = materialize(Path(tmp), not UNPATCHED)
            spec = importlib.util.spec_from_file_location("audio_depthwise_gate", target)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            calls = []
            kernel = module._depthwise_conv1d_kernel

            def record_kernel(**kwargs):
                calls.append(True)
                return kernel(**kwargs)

            module._depthwise_conv1d_kernel = record_kernel
            for dtype, channels, width, stride, padding in (
                (np.float32, 3, 3, 1, 1),
                (np.float16, 2, 5, 1, 2),
                (np.float32, 3, 3, 2, 0),
            ):
                x = (((np.arange(18 * channels) % 11) - 5) / 8).astype(dtype).reshape(2, 9, channels)
                w = (((np.arange(channels * width) % 5) - 2) / 16).astype(dtype).reshape(channels, width, 1)
                length = (x.shape[1] + 2 * padding - width) // stride + 1
                want = np.zeros((2, length, channels), dtype=dtype)
                for b in range(2):
                    for t in range(length):
                        for c in range(channels):
                            want[b, t, c] = sum(float(x[b, i, c]) * float(w[c, k, 0])
                                                for k in range(width)
                                                if 0 <= (i := t * stride + k - padding) < x.shape[1])
                got = module.depthwise_conv1d(mx.array(x), mx.array(w), stride=stride,
                                              padding=padding, groups=channels)
                mx.eval(got)
                np.testing.assert_array_equal(np.array(got), want)
                self.assertEqual(calls, [], "custom Metal route selected with Metal unavailable")


if __name__ == "__main__":
    unittest.main()
