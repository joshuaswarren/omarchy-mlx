"""Contract tests for the greedy sampler identity (patch 0010).

oMLX built its temp == 0 sampler as an anonymous ``lambda x: mx.argmax(x,
axis=-1)``, so the mlx-omarchy BatchGenerator (scripts/patch-mlx-lm-batch-
greedy-head.py) could not tell a greedy batch from a sampled one and always
projected every row onto the full vocabulary. With the patch, temp == 0
returns mlx-lm's own ``greedy_sampler`` (plain argmax, no mx.compile), which
the BatchGenerator recognises by identity.
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
    from mlx_lm.sample_utils import greedy_sampler
    from omlx.utils.sampling import make_sampler
except ImportError as exc:  # pragma: no cover - dev boxes without the mlx wheel
    raise unittest.SkipTest(f"omlx.utils.sampling unavailable: {exc}")


class GreedySamplerIdentity(unittest.TestCase):
    def test_temp_zero_is_mlx_lm_greedy_sampler(self):
        self.assertIs(make_sampler(temp=0.0), greedy_sampler)

    def test_greedy_picks_the_argmax_row_by_row(self):
        logprobs = mx.array([[0.1, 2.0, -1.0], [3.0, 0.0, 2.5]])
        self.assertEqual(make_sampler(temp=0.0)(logprobs).tolist(), [1, 0])

    def test_sampling_temperatures_are_not_greedy(self):
        self.assertIsNot(make_sampler(temp=0.7), greedy_sampler)


if __name__ == "__main__":
    unittest.main()
