"""Contract tests for the greedy sampler tag (patch 0010).

oMLX builds its temp == 0 sampler as its own argmax lambda, which the
mlx-omarchy BatchGenerator (scripts/patch-mlx-lm-batch-greedy-head.py)
cannot tell from a sampled one. The patch tags that lambda
``_mlx_omarchy_greedy = True``. It must stay a fresh per-request object:
``make_sampler`` sets ``temp``/``top_p``/``top_k``/... on the sampler it
returns, so handing out one shared global would let every temp-0 request
write onto the same object.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

OMLX_SRC = Path(os.environ.get("OMLX_SRC", "/tmp/omlx-applied"))
sys.path.insert(0, str(OMLX_SRC))

try:
    import mlx.core as mx
    from omlx.utils.sampling import make_sampler
except ImportError as exc:  # pragma: no cover - dev boxes without the mlx wheel
    raise unittest.SkipTest(f"omlx.utils.sampling unavailable: {exc}")


class GreedySamplerTag(unittest.TestCase):
    def test_temp_zero_is_tagged(self):
        self.assertTrue(getattr(make_sampler(temp=0.0), "_mlx_omarchy_greedy", False))

    def test_each_request_gets_its_own_sampler(self):
        self.assertIsNot(make_sampler(temp=0.0), make_sampler(temp=0.0))

    def test_greedy_picks_the_argmax_row_by_row(self):
        logprobs = mx.array([[0.1, 2.0, -1.0], [3.0, 0.0, 2.5]])
        self.assertEqual(make_sampler(temp=0.0)(logprobs).tolist(), [1, 0])

    def test_sampling_temperatures_are_not_tagged(self):
        self.assertFalse(getattr(make_sampler(temp=0.7), "_mlx_omarchy_greedy", False))


if __name__ == "__main__":
    unittest.main()
