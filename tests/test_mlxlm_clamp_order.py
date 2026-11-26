"""Source-order guard: the uclamp clamp must run before mlx starts its threads.

Run: python3 -m unittest tests.test_mlxlm_clamp_order -q

Same pattern as ClampOrderTests in tests/test_bench_decode.py: MLX spawns
its stream threads at import, so apply_from_env() in _mlxlm_server.main()
must execute before `import mlx_lm.server`, or decode runs at the stock
clamp (Oct 16 miss).
"""

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER = REPO_ROOT / "serve" / "mlx_omarchy_serve" / "_mlxlm_server.py"


class ClampOrderTests(unittest.TestCase):
    def test_clamp_runs_before_mlx_lm_server_is_imported(self):
        src = SERVER.read_text()
        clamp = src.index("apply_from_env()")
        self.assertLess(clamp, src.index("import mlx_lm.server"))


if __name__ == "__main__":
    unittest.main()
