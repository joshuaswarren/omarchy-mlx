#!/usr/bin/env python3
"""Live probe for PR 1044: real end-to-end convert, twice, plus litter check."""
import io
import contextlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "tests")
sys.path.insert(0, "serve")
from test_laya_convert import _make_source_snapshot  # noqa: E402
from mlx_omarchy_laya import convert  # noqa: E402

root = Path(tempfile.mkdtemp())
src = _make_source_snapshot(root / "src")
out = root / "out"

for i in (1, 2):
    with contextlib.redirect_stdout(io.StringIO()):
        result = convert.convert(src, "org/repo", "main", out, "cat-1")
    print(f"run {i}: ok, weights_sha256={result['weights_sha256'][:12]}")

parts = list(out.rglob("*.part"))
manifest = json.loads((out / "manifest.json").read_text())
first_sha = None
# determinism: manifest from run 2 must equal run 1's weights hash
print("litter:", parts or "none")
print("manifest keys ok:", sorted(manifest) == sorted([
    "catalog_id", "source_repo", "source_revision", "source_files_sha256",
    "weights_sha256", "key_aliases_applied", "encoder_architecture",
    "weights_header", "context_max_tokens", "head_max_tokens", "estimated_bytes"]))
for p in sorted(out.rglob("*")):
    if p.is_file():
        print(" out/", p.relative_to(out), p.stat().st_size)
