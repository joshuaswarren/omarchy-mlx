"""Dispatch-mode profiler stamps must ride the dispatch's own stream.

Issue omarchy-mlx#526: on the Apple M6 (G18G B1) the per-dispatch
diagnostics build dies in prefill with VK_ERROR_DEVICE_LOST. Borealis
attaches BOTTOM_OF_PIPE timestamps to an existing stream's tail; a stamp
recorded after the previous dispatch's t1 (with the dependency barrier
skipped) cannot ride a stream that already carries an end timestamp, so
it falls back to timestamp_only_stream() -> stamp_carrier_launch(), which
no-ops without agx_native_graphics(). The command-free stream reaches the
kernel with stats.cmds == 0 -> -EINVAL -> device-lost.

The contract tested here: before_dispatch records t0 at TOP_OF_PIPE so
the driver parks it on pending_start_ts and attaches it to the START of
the dispatch's own command-bearing stream, while after_dispatch keeps t1
at BOTTOM_OF_PIPE so it rides the same stream's tail. No timestamp write
in dispatch mode may sit at BOTTOM_OF_PIPE in a position that can orphan
onto a timestamp-only stream.

Stdlib unittest only; parses the header like the other source-contract
tests in this directory.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADER = REPO_ROOT / "overlay" / "mlx" / "backend" / "omarchy" / "gpu_profiler.h"


def _fn_body(src: str, name: str) -> str:
    """Return the body of a `name(` member function in the header."""
    m = re.search(rf"\b{name}\s*\(", src)
    assert m, f"{name} not found in gpu_profiler.h"
    start = src.index("{", m.start())
    depth, i = 0, start
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i]
        i += 1


class TestDispatchTimestampStages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = HEADER.read_text()

    def test_t0_uses_top_of_pipe(self):
        body = _fn_body(self.src, "before_dispatch")
        writes = re.findall(r"VK_PIPELINE_STAGE_(\w+)_BIT", body)
        self.assertIn(
            "TOP_OF_PIPE",
            writes,
            "dispatch t0 must be TOP_OF_PIPE so pending_start_ts attaches it "
            "to the dispatch's own stream (a bare BOTTOM_OF_PIPE write after "
            "the previous t1 orphans onto a command-free stream -> EINVAL)",
        )
        self.assertNotIn(
            "BOTTOM_OF_PIPE",
            writes,
            "a BOTTOM_OF_PIPE t0 after the previous dispatch's t1 is the "
            "standalone stamp that kills the M6 submit",
        )

    def test_t1_stays_bottom_of_pipe(self):
        body = _fn_body(self.src, "after_dispatch")
        self.assertIn(
            "VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT",
            body,
            "t1 must ride the tail of the dispatch's own stream",
        )

    def test_barrier_still_recorded_in_before_dispatch(self):
        body = _fn_body(self.src, "before_dispatch")
        self.assertIn(
            "CmdPipelineBarrier",
            body,
            "the pre-dispatch barrier stays unconditional (Neon's scoped fix "
            "keeps the t0/t1 isolation guarantee)",
        )


if __name__ == "__main__":
    unittest.main()
