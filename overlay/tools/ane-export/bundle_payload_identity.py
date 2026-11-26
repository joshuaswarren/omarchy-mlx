"""Canonical identity for a compiled payload collection."""

import hashlib
import json


def payload_collection_sha256(payloads: list[dict]) -> str:
    records = [
        {
            "role": payload["role"],
            "path": payload["path"],
            "byte_size": payload["byte_size"],
            "sha256": payload["sha256"],
        }
        for payload in sorted(payloads, key=lambda payload: payload["path"])
    ]
    encoded = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
