# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Receipt JSON handling in the Parakeet TDT capture comparison CLI."""

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml import parakeet_tdt


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


# The device modules compare_vulkan_capture imports; none of their callables
# runs before the parent receipt check.
DEVICE_STUBS = {
    "mlx": _module("mlx"),
    "mlx.core": _module("mlx.core"),
    "coreml.vulkan_decoder": _module("coreml.vulkan_decoder", load_decoder=None),
    "coreml.vulkan_joint": _module("coreml.vulkan_joint", run_joint=None),
    "coreml.vulkan_mel": _module("coreml.vulkan_mel", trace_snapshot=None),
}


class ReceiptJsonTest(unittest.TestCase):
    def test_compare_rejects_unauthenticated_parent_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture.mkdir()
            (capture / "manifest.sha256").write_text("")
            (root / "receipt.json").write_text("{}")
            with (
                mock.patch.dict(sys.modules, DEVICE_STUBS),
                self.assertRaisesRegex(
                    parakeet_tdt.TdtControlError,
                    "parent receipt does not authenticate the capture manifest",
                ),
            ):
                parakeet_tdt.compare_vulkan_capture(
                    capture,
                    root / "model",
                    source_commit="0" * 40,
                    runtime_source_commit="1" * 40,
                )

    def test_main_prints_and_writes_the_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "receipt.json"
            argv = ["capture", "model", "--source-commit", "a", "--runtime-source-commit",
                    "b", "--output", str(output)]
            stdout = io.StringIO()
            with (
                mock.patch.object(parakeet_tdt, "compare_vulkan_capture",
                                  return_value={"execution": {}}),
                contextlib.redirect_stdout(stdout),
            ):
                self.assertEqual(parakeet_tdt.main(argv), 0)
            self.assertEqual(output.read_text(), stdout.getvalue())
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {"execution": {"command": [sys.executable, "-m", "coreml.parakeet_tdt", *argv]}},
        )


if __name__ == "__main__":
    unittest.main()
