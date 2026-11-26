# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Hermetic tests for the fetch_parakeet_reference downloader CLI.

A local HTTP server impersonates the two Hugging Face endpoints the
downloader uses (tree API + resolve URLs) with tiny synthetic files,
so missing/tampered/cached behavior is exercised without 458 MB of
weights. The real-cache command runs as an integration test when the
model cache is present.
"""

import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "overlay" / "tools" / "coreml" / "fetch_parakeet_reference.py"

MODEL_REPO = "mweinbach1/parakeet-tdt-0.6b-v3-coreml"
REVISION = "b650695c2322ee5281dff48d7345b2f3a58ff018"

# tokenizer.json pretends to be LFS (sha256 pinned as lfs oid);
# Manifest.json pretends to be a plain git blob file (oid = blob sha1).
_MANIFEST = b'{"fake": "manifest"}'
_MLMODEL = b'{"fake": "spec"}'
FILES = {
    "tokenizer.json": b'{"fake": "tokenizer"}',
    "encoder.mlpackage/Manifest.json": _MANIFEST,
    "encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel": _MLMODEL,
    "decoder.mlpackage/Manifest.json": _MANIFEST,
    "decoder.mlpackage/Data/com.apple.CoreML/model.mlmodel": _MLMODEL,
    "joint.mlpackage/Manifest.json": _MANIFEST,
    "joint.mlpackage/Data/com.apple.CoreML/model.mlmodel": _MLMODEL,
}
LFS_PATHS = {p for p in FILES if not p.endswith("Manifest.json")}


def blob_id(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def tree_payload(file_bytes: dict[str, bytes]) -> list[dict]:
    entries = []
    for p in sorted(file_bytes):
        b = file_bytes[p]
        if p in LFS_PATHS:
            entries.append({"type": "file", "path": p, "size": len(b),
                            "lfs": {"oid": hashlib.sha256(b).hexdigest(),
                                    "size": len(b)}})
        else:
            entries.append({"type": "file", "path": p, "size": len(b),
                            "oid": blob_id(b)})
    return entries


class FakeHF(ThreadingHTTPServer):
    """Serves HF endpoint shapes; ``served_bytes`` may lie about content."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.file_bytes = dict(FILES)      # canonical tree source
        self.served_bytes = dict(FILES)    # what resolve actually returns
        self.resolve_hits = 0

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class Handler(BaseHTTPRequestHandler):
    server: FakeHF

    def log_message(self, *args):  # silence
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if f"/api/models/{MODEL_REPO}/tree/{REVISION}" in self.path:
            self._send(200, json.dumps(tree_payload(self.server.file_bytes)).encode(),
                       "application/json")
            return
        marker = f"/{MODEL_REPO}/resolve/{REVISION}/"
        if marker in self.path:
            path = self.path.split(marker, 1)[1].split("?")[0]
            data = self.server.served_bytes.get(path)
            if data is None:
                self._send(404, b"nope", "text/plain")
            else:
                self.server.resolve_hits += 1
                self._send(200, data, "application/octet-stream")
            return
        self._send(404, b"nope", "text/plain")


def synthetic_lock():
    sys.path.insert(0, str(CLI.parent))
    try:
        import reference as R
        return R.ReferenceLock(
            schema_version=1,
            reference_repo="mweinbach/parakeet-coreml-swift",
            reference_commit="a" * 40,
            model_repo=MODEL_REPO,
            model_revision=REVISION,
            model_license="CC-BY-4.0",
            model_quantization="synthetic",
            files=[
                R.LockedFile(path=p, size=len(b),
                             sha256=hashlib.sha256(b).hexdigest())
                for p, b in sorted(FILES.items())
            ],
            audio=R.AudioFixture(
                url="https://example.invalid/jfk.flac", sha256="c" * 64,
                size=1, sample_rate=44100, duration_seconds=11.0,
                license="CC-BY-4.0", note="synthetic",
            ),
            mel=R.MelConfig(sample_rate=16000, hop_length=160, win_length=400,
                            n_fft=512, n_mels=128, preemphasis=0.97,
                            log_guard=2.0**-24, epsilon=1e-5),
            tdt=R.TdtConfig(blank_token_id=8192, durations=[0, 1, 2, 3, 4],
                            max_symbols_per_step=10, vocab_size=8193),
            numerical_contract=None, macos_reference_environment=None,
            macos_reference_paths={},
        )
    finally:
        sys.path.remove(str(CLI.parent))


@pytest.fixture()
def fake_hf(tmp_path):
    server = FakeHF()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sys.path.insert(0, str(CLI.parent))
    try:
        import reference as R
        lock_path = tmp_path / "synthetic.lock"
        R.write_lock(synthetic_lock(), lock_path)
    finally:
        sys.path.remove(str(CLI.parent))
    yield server, tmp_path / "cache-root", lock_path
    server.shutdown()


def run_cli(*args, cache_root, endpoint, lock, expect=0):
    env = dict(os.environ)
    if endpoint:
        env["MLX_OMARCHY_HF_ENDPOINT"] = endpoint
    proc = subprocess.run(
        [sys.executable, str(CLI), *args, "--cache-root", str(cache_root),
         "--lock", str(lock)],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == expect, (
        f"args={args} rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    return proc


def cache_dir_for(cache_root: Path) -> Path:
    return cache_root / "parakeet-reference" / MODEL_REPO / REVISION


# ---------- hermetic CLI tests ----------


def test_download_fresh_then_cached_no_refetch(fake_hf):
    server, cache_root, lock = fake_hf
    run_cli("download", cache_root=cache_root, endpoint=server.endpoint, lock=lock)
    cdir = cache_dir_for(cache_root)
    assert (cdir / "tokenizer.json").read_bytes() == FILES["tokenizer.json"]
    assert server.resolve_hits == len(FILES)

    # second run must not touch the network: dead server, still succeeds
    server.shutdown()
    run_cli("download", cache_root=cache_root, endpoint=server.endpoint, lock=lock)
    assert server.resolve_hits == len(FILES)  # unchanged


def test_download_repairs_tampered_local_file(fake_hf):
    server, cache_root, lock = fake_hf
    run_cli("download", cache_root=cache_root, endpoint=server.endpoint, lock=lock)
    cdir = cache_dir_for(cache_root)
    (cdir / "tokenizer.json").write_bytes(b"tampered-content-xxxx")  # same size, 21 bytes
    proc = run_cli("verify", cache_root=cache_root, endpoint=server.endpoint,
                   lock=lock, expect=1)
    assert "sha256-mismatch: tokenizer.json" in proc.stderr
    run_cli("download", cache_root=cache_root, endpoint=server.endpoint, lock=lock)
    assert (cdir / "tokenizer.json").read_bytes() == FILES["tokenizer.json"]
    run_cli("verify", cache_root=cache_root, endpoint=server.endpoint, lock=lock)

def test_download_missing_file_refetched(fake_hf):
    server, cache_root, lock = fake_hf
    run_cli("download", cache_root=cache_root, endpoint=server.endpoint, lock=lock)
    cdir = cache_dir_for(cache_root)
    hits = server.resolve_hits
    (cdir / "joint.mlpackage" / "Manifest.json").unlink()
    run_cli("download", cache_root=cache_root, endpoint=server.endpoint, lock=lock)
    assert server.resolve_hits == hits + 1
    run_cli("verify", cache_root=cache_root, endpoint=server.endpoint, lock=lock)


def test_download_refuses_upstream_drift(fake_hf):
    """Upstream revision content moved (tree no longer matches pin)."""
    server, cache_root, lock = fake_hf
    run_cli("download", cache_root=cache_root, endpoint=server.endpoint, lock=lock)
    server.file_bytes["tokenizer.json"] = b'{"moved": "upstream"}'
    proc = run_cli("download", "--force", cache_root=cache_root,
                   endpoint=server.endpoint, lock=lock, expect=2)
    assert "upstream content drift" in proc.stderr


def test_download_refuses_corrupt_transfer(fake_hf):
    """Served bytes differ from the pinned tree: refuse to use them."""
    server, cache_root, lock = fake_hf
    corrupt = bytes(reversed(FILES["tokenizer.json"]))
    assert len(corrupt) == len(FILES["tokenizer.json"])
    server.served_bytes["tokenizer.json"] = corrupt
    proc = run_cli("download", cache_root=cache_root,
                   endpoint=server.endpoint, lock=lock, expect=2)
    assert "does not match the pin" in proc.stderr
    # the refused content must not verify: fail closed on the next check
    proc = run_cli("verify", cache_root=cache_root,
                   endpoint=server.endpoint, lock=lock, expect=1)
    assert "sha256-mismatch: tokenizer.json" in proc.stderr


def test_path_command_is_predictable(fake_hf):
    server, cache_root, lock = fake_hf
    proc = run_cli("path", cache_root=cache_root, endpoint=server.endpoint, lock=lock)
    assert proc.stdout.strip() == str(cache_dir_for(cache_root))


def test_info_reports_cache_state(fake_hf):
    server, cache_root, lock = fake_hf
    info = json.loads(run_cli("info", cache_root=cache_root,
                              endpoint=server.endpoint, lock=lock).stdout)
    assert info["cache_ok"] is False
    assert info["golden_captured"] is False
    run_cli("download", cache_root=cache_root, endpoint=server.endpoint, lock=lock)
    info = json.loads(run_cli("info", cache_root=cache_root,
                              endpoint=server.endpoint, lock=lock).stdout)
    assert info["cache_ok"] is True


# ---------- real-cache integration ----------


def test_real_cache_download_and_verify_commands():
    """Real pinned cache + real lock + real HF tree, no fake server."""
    real_cache = (
        Path.home() / ".cache/mlx-omarchy/parakeet-reference"
        / MODEL_REPO / REVISION
    )
    if not real_cache.is_dir():
        pytest.skip("real model cache not present on this host")
    real_lock = CLI.parent / "parakeet-reference.lock"
    dl = subprocess.run(
        [sys.executable, str(CLI), "download", "--lock", str(real_lock)],
        capture_output=True, text=True,
    )
    assert dl.returncode == 0, dl.stdout + dl.stderr
    assert "cache already verifies" in dl.stdout
    vf = subprocess.run(
        [sys.executable, str(CLI), "verify", "--lock", str(real_lock)],
        capture_output=True, text=True,
    )
    assert vf.returncode == 0, vf.stdout + vf.stderr
    assert "OK: 12 files verified" in vf.stdout
