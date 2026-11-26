# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Integrity-checked host detokenization for the pinned Parakeet tokenizer."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_METASPACE = "▁"


class TokenizerError(RuntimeError):
    """The tokenizer file cannot be loaded or has an unsupported format."""


class TokenizerIntegrityError(TokenizerError):
    """The tokenizer content does not match its expected SHA-256 identity."""


@dataclass(frozen=True)
class ParakeetTokenizer:
    """The minimum Hugging Face BPE detokenizer used by Parakeet TDT."""

    id_to_piece: tuple[str, ...]
    special_ids: frozenset[int]
    sha256: str

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_piece)

    @classmethod
    def load(
        cls, path: str | Path, *, expected_sha256: str
    ) -> ParakeetTokenizer:
        """Load only content matching ``expected_sha256`` and verify its schema."""
        if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(
            expected_sha256
        ):
            raise TokenizerIntegrityError(
                "expected tokenizer sha256 must be 64 lowercase hex characters"
            )

        tokenizer_path = Path(path)
        try:
            raw = tokenizer_path.read_bytes()
        except OSError as error:
            raise TokenizerError(
                f"could not read tokenizer {tokenizer_path}: {error}"
            ) from error

        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != expected_sha256:
            raise TokenizerIntegrityError(
                f"tokenizer sha256 mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )

        try:
            root = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TokenizerError(f"tokenizer is not valid UTF-8 JSON: {error}") from error

        pieces, special_ids = _parse_tokenizer(root)
        return cls(tuple(pieces), frozenset(special_ids), actual_sha256)

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = True) -> str:
        """Decode IDs with the pinned Metaspace behavior used by the Swift reference."""
        parts: list[str] = []
        for token_id in token_ids:
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or token_id >= len(self.id_to_piece)
            ):
                continue
            if skip_special and token_id in self.special_ids:
                continue
            parts.append(self.id_to_piece[token_id].replace(_METASPACE, " "))
        text = "".join(parts)
        return text[1:] if text.startswith(" ") else text


def _parse_tokenizer(root: object) -> tuple[list[str], set[int]]:
    if not isinstance(root, dict):
        raise TokenizerError("tokenizer root must be a JSON object")
    if root.get("version") != "1.0":
        raise TokenizerError("tokenizer version must be '1.0'")

    decoder = root.get("decoder")
    if not isinstance(decoder, dict) or (
        decoder.get("type") != "Metaspace"
        or decoder.get("replacement") != _METASPACE
        or decoder.get("prepend_scheme") != "always"
        or decoder.get("split") is not True
    ):
        raise TokenizerError(
            "tokenizer must use the Metaspace decoder with replacement '▁', "
            "prepend_scheme 'always', and split true"
        )

    model = root.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise TokenizerError("tokenizer model must be a BPE object")
    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise TokenizerError("tokenizer model.vocab must be a non-empty object")

    pairs: list[tuple[str, int, bool]] = []
    for piece, token_id in vocab.items():
        _validate_pair(piece, token_id, "model.vocab")
        pairs.append((piece, token_id, False))

    added_tokens = root.get("added_tokens", [])
    if not isinstance(added_tokens, list):
        raise TokenizerError("tokenizer added_tokens must be an array")
    for index, token in enumerate(added_tokens):
        if not isinstance(token, dict):
            raise TokenizerError(f"added token {index} must be an object")
        piece = token.get("content")
        token_id = token.get("id")
        special = token.get("special", False)
        _validate_pair(piece, token_id, f"added token {index}")
        if not isinstance(special, bool):
            raise TokenizerError(f"added token {index} special flag must be boolean")
        pairs.append((piece, token_id, special))

    by_id: dict[int, str] = {}
    special_ids: set[int] = set()
    for piece, token_id, explicit_special in pairs:
        previous = by_id.get(token_id)
        if previous is not None and previous != piece:
            raise TokenizerError(
                f"token id {token_id} maps to both {previous!r} and {piece!r}"
            )
        by_id[token_id] = piece
        if explicit_special or (piece.startswith("<") and piece.endswith(">")):
            special_ids.add(token_id)

    max_id = max(by_id)
    id_to_piece = [""] * (max_id + 1)
    for token_id, piece in by_id.items():
        id_to_piece[token_id] = piece
    return id_to_piece, special_ids


def _validate_pair(piece: object, token_id: object, label: str) -> None:
    if not isinstance(piece, str):
        raise TokenizerError(f"{label} piece must be a string")
    if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
        raise TokenizerError(f"{label} id must be a non-negative integer")
