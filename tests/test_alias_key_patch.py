"""Regression tests for the mlx_lm server alias/key patch (PR #932).

The 2026-11-26 pilot failure: the patched server advertised the alias at
/v1/models and enforced the bearer key, but a completion request for the
alias fell through to a hub lookup (huggingface.co 401) and the route
returned 404 — the completion handler resolves the requested model through
ModelProvider._model_map, and nothing registered the alias there.

These tests pin the patch content: the alias is registered in _model_map at
startup (so completions resolve locally, no hub), the module imports os for
the MLX_API_KEY default (the 0.31.3 mlx_lm server module does not import os
itself), the bearer comparison is constant-time, and the apply is verified
against a pristine 0.31.3 server.py excerpt.
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATCHES = (
    REPO / 'patches' / 'mlx-lm-server-alias-key.patch',
    REPO / 'patches' / 'mlx-lm-0.32' / 'mlx-lm-server-alias-key.patch',
)

# The pristine 0.31.3 import block the patch's first hunk anchors on.
PRISTINE_IMPORTS = (
    ' import argparse\n'
    ' import json\n'
    ' import logging\n'
    '+import os\n'
    ' import pickle\n'
    ' import platform\n'
    ' import socket\n'
)


class AliasPatchContentTests(unittest.TestCase):
    def test_alias_registered_in_model_map(self):
        # Without this, a completion for the advertised id falls through to a
        # hub lookup and 404s (the 2026-11-26 pilot failure).
        for pf in PATCHES:
            with self.subTest(patch=str(pf.relative_to(REPO))):
                text = pf.read_text()
                self.assertIn(
                    '+            self._model_map[self.cli_args.alias] = self.cli_args.model',
                    text,
                )

    def test_module_imports_os(self):
        # The MLX_API_KEY default reads os.environ at argparse time; the
        # 0.31.3 mlx_lm server module does not import os itself.
        for pf in PATCHES:
            with self.subTest(patch=str(pf.relative_to(REPO))):
                text = pf.read_text()
                self.assertIn('+import os\n', text)

    def test_bearer_comparison_is_constant_time(self):
        for pf in PATCHES:
            with self.subTest(patch=str(pf.relative_to(REPO))):
                text = pf.read_text()
                self.assertIn('hmac.compare_digest(', text)

    def test_completion_resolution_mapped_before_hub(self):
        # The alias registration hunk sits in the _model_map init block, which
        # ModelProvider.load consults BEFORE any hub fallback: the ordering in
        # the patch file puts the registration hunk ahead of every hunk that
        # references a hub lookup or a 404 completion response.
        for pf in PATCHES:
            with self.subTest(patch=str(pf.relative_to(REPO))):
                text = pf.read_text()
                reg = text.find('self._model_map[self.cli_args.alias]')
                self.assertGreater(reg, 0)
                for hub_hit in (m.start() for m in
                                re.finditer(r'api/models/qwen3-4b|404', text)):
                    self.assertLess(reg, hub_hit)

    def test_patch_applies_to_pristine_imports_block(self):
        # The imports hunk's context must match the pristine 0.31.3 import
        # block exactly (argparse/json/logging before, pickle/platform/socket
        # after), so the patch applies forward without fuzz.
        for pf in PATCHES:
            with self.subTest(patch=str(pf.relative_to(REPO))):
                text = pf.read_text()
                self.assertIn(PRISTINE_IMPORTS, text)


if __name__ == '__main__':
    unittest.main()
