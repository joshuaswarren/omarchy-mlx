# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Golden and boundary tests for pinned Parakeet detokenization."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.tokenizer import ParakeetTokenizer, TokenizerError, TokenizerIntegrityError

_FIXTURES = Path(__file__).with_name("fixtures")
_TDT_FIXTURE = _FIXTURES / "tdt_control_golden.json"
_TOKENIZER_FIXTURE = _FIXTURES / "tokenizer_golden.json"
_MODEL_REPO = "mweinbach1/parakeet-tdt-0.6b-v3-coreml"


def _cache_root(revision: str) -> Path | None:
    base = os.environ.get("MLX_OMARCHY_CACHE_DIR")
    roots = []
    if base:
        roots.append(Path(base) / "parakeet-reference")
    roots.append(Path.home() / ".cache" / "mlx-omarchy" / "parakeet-reference")
    for root in roots:
        candidate = root / _MODEL_REPO / revision
        if candidate.is_dir():
            return candidate
    return None


def _write_tokenizer(path: Path, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class TokenizerTest(unittest.TestCase):
    def test_pinned_tokens_decode_to_exact_reference_transcript(self):
        golden = json.loads(_TOKENIZER_FIXTURE.read_text())
        cache = _cache_root(golden["model_revision"])
        if cache is None:
            self.skipTest("parakeet reference cache not present")
        tdt_golden = json.loads(_TDT_FIXTURE.read_text())
        self.assertEqual(
            tdt_golden["source"]["token_output_sha256"],
            golden["token_output_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(golden["transcript"].encode()).hexdigest(),
            golden["transcript_sha256"],
        )

        tokenizer = ParakeetTokenizer.load(
            cache / "tokenizer.json",
            expected_sha256=golden["tokenizer_sha256"],
        )

        self.assertEqual(tokenizer.sha256, golden["tokenizer_sha256"])
        self.assertEqual(tokenizer.vocab_size, 8193)
        self.assertEqual(
            tokenizer.decode(tdt_golden["expected"]["token_ids"]),
            golden["transcript"],
        )

    def test_integrity_failure_precedes_json_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            path.write_bytes(b"not json")

            with self.assertRaisesRegex(TokenizerIntegrityError, "sha256 mismatch"):
                ParakeetTokenizer.load(path, expected_sha256="0" * 64)

    def test_metaspace_special_and_unknown_token_semantics(self):
        payload = {
            "version": "1.0",
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": True},
                {"id": 3, "content": "<blank>", "special": True},
            ],
            "decoder": {
                "type": "Metaspace",
                "replacement": "▁",
                "prepend_scheme": "always",
                "split": True,
            },
            "model": {
                "type": "BPE",
                "vocab": {"<unk>": 0, "▁hello": 1, "▁world": 2},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            digest = _write_tokenizer(path, payload)
            tokenizer = ParakeetTokenizer.load(path, expected_sha256=digest)

        self.assertEqual(tokenizer.decode((0, 1, 2, 3, -1, 100)), "hello world")
        self.assertEqual(tokenizer.decode((0, 1), skip_special=False), "<unk> hello")

    def test_unsupported_decoder_format_is_named(self):
        payload = {
            "version": "1.0",
            "added_tokens": [],
            "decoder": {"type": "ByteLevel"},
            "model": {"type": "BPE", "vocab": {"▁hello": 0}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            digest = _write_tokenizer(path, payload)

            with self.assertRaisesRegex(TokenizerError, "Metaspace decoder"):
                ParakeetTokenizer.load(path, expected_sha256=digest)


if __name__ == "__main__":
    unittest.main()
