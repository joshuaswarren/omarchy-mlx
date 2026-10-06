#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the DeepFilterNet local-path subfolder rule (patch 0009).

mlx-audio's ``DeepFilterNetModel.from_pretrained`` appends its default
``v3`` subfolder to every local path, so a locally hosted model directory
that already IS one version's content (``config.json`` + ``model.safetensors``
at the top, exactly what oMLX discovery serves) resolves to
``<dir>/v3/config.json`` and fails with ``Missing config.json``. Only the
HuggingFace repo id keeps the version subfolders.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

OMLX_SRC = Path(os.environ.get("OMLX_SRC", "/tmp/omlx-applied"))
sys.path.insert(0, str(OMLX_SRC))

try:
    from omlx.engine.sts import _deepfilternet_subfolder
except ImportError as exc:  # pragma: no cover - dev boxes without the mlx wheel
    raise unittest.SkipTest(f"omlx.engine.sts unavailable: {exc}")

FIXTURE = Path(__file__).resolve().parent / "fixtures/deepfilternet3-config.json"


class DeepFilterNetSubfolder(unittest.TestCase):
    def test_hosted_version_dir_gets_no_subfolder(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "deepfilter"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(FIXTURE.read_text())

            self.assertIsNone(_deepfilternet_subfolder(str(model_dir)))

    def test_parent_layout_keeps_default_subfolder(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "deepfilter"
            model_dir.mkdir()

            self.assertEqual(_deepfilternet_subfolder(str(model_dir)), "v3")


if __name__ == "__main__":
    unittest.main()
