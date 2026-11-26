# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Tests for the parakeet-reference lock schema and cache verification."""

import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools" / "coreml"
sys.path.insert(0, str(TOOLS))

import reference as R


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make_lock(**overrides) -> R.ReferenceLock:
    kwargs = {
        "schema_version": 1,
        "reference_repo": "mweinbach/parakeet-coreml-swift",
        "reference_commit": "a" * 40,
        "model_repo": "mweinbach1/parakeet-tdt-0.6b-v3-coreml",
        "model_revision": "b" * 40,
        "model_license": "CC-BY-4.0",
        "model_quantization": "test",
        "files": [
            R.LockedFile(path=p, size=len(b), sha256=sha256(b))
            for p, b in sorted(SYNTHETIC_FILES.items())
        ],
        "audio": R.AudioFixture(
            url="https://example.invalid/jfk.flac",
            sha256="c" * 64,
            size=1,
            sample_rate=44100,
            duration_seconds=11.0,
            license="CC-BY-4.0",
            note="test",
        ),
        "mel": R.MelConfig(
            sample_rate=16000, hop_length=160, win_length=400, n_fft=512,
            n_mels=128, preemphasis=0.97, log_guard=2.0**-24, epsilon=1e-5,
        ),
        "tdt": R.TdtConfig(
            blank_token_id=8192, durations=[0, 1, 2, 3, 4],
            max_symbols_per_step=10, vocab_size=8193,
        ),
        "numerical_contract": None,
        "macos_reference_environment": None,
        "macos_reference_paths": {},
    }
    kwargs.update(overrides)
    return R.ReferenceLock(**kwargs)


SYNTHETIC_FILES = {
    "tokenizer.json": b'{"hello": "tokenizer"}',
    "joint.mlpackage/Manifest.json": b'{"manifest": true}',
}


def write_cache(cache_dir: Path, files: dict[str, bytes]) -> None:
    for p, b in files.items():
        f = cache_dir / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b)


def small_cache(tmp_path: Path):
    """Synthetic cache that skips the package-completeness rule."""
    lock = make_lock()
    lock.files = [
        R.LockedFile(path=p, size=len(b), sha256=sha256(b))
        for p, b in sorted(SYNTHETIC_FILES.items())
    ]
    cache = tmp_path / "cache"
    write_cache(cache, SYNTHETIC_FILES)
    return lock, cache


# ---------- lock schema ----------


def test_real_lock_loads_and_validates():
    lock = R.ReferenceLock.load(TOOLS / "parakeet-reference.lock")
    R.validate_lock(lock)
    assert lock.schema_version == 1
    assert lock.model_repo == "mweinbach1/parakeet-tdt-0.6b-v3-coreml"
    assert lock.reference_repo == "mweinbach/parakeet-coreml-swift"
    assert lock.reference_commit.startswith("75aec2a")
    assert lock.model_revision.startswith("b650695")
    paths = {f.path for f in lock.files}
    assert "tokenizer.json" in paths
    for pkg in ("encoder", "decoder", "joint"):
        assert f"{pkg}.mlpackage/Manifest.json" in paths
        assert f"{pkg}.mlpackage/Data/com.apple.CoreML/model.mlmodel" in paths
        assert f"{pkg}.mlpackage/Data/com.apple.CoreML/weights/weight.bin" in paths
    assert lock.mel.n_mels == 128 and lock.tdt.blank_token_id == 8192


def test_lock_roundtrip():
    lock = R.ReferenceLock.load(TOOLS / "parakeet-reference.lock")
    clone = R.ReferenceLock.from_dict(lock.to_dict())
    assert clone.to_dict() == lock.to_dict()


def test_real_lock_matches_real_cache_when_present():
    cache = (
        R.default_cache_root()
        / "mweinbach1/parakeet-tdt-0.6b-v3-coreml"
        / "b650695c2322ee5281dff48d7345b2f3a58ff018"
    )
    if not cache.is_dir():
        pytest.skip("real model cache not present on this host")
    lock = R.ReferenceLock.load(TOOLS / "parakeet-reference.lock")
    ok, mismatches = R.verify_cache(cache, lock)
    assert ok, mismatches


@pytest.mark.parametrize(
    "field, value",
    [
        ("schema_version", 99),
        ("reference_commit", "xyz"),
        ("model_revision", "a" * 39),
        ("files", []),
    ],
)
def test_lock_validation_rejects(field, value):
    lock = make_lock(**{field: value})
    with pytest.raises(R.ReferenceError):
        R.validate_lock(lock)


def test_lock_validation_rejects_bad_audio_hash():
    lock = make_lock(audio=replace(make_lock().audio, sha256="c" * 63))
    with pytest.raises(R.ReferenceError):
        R.validate_lock(lock)


def test_lock_validation_rejects_blank_outside_vocab():
    base = make_lock()
    lock = make_lock(tdt=replace(base.tdt, blank_token_id=9999))
    with pytest.raises(R.ReferenceError):
        R.validate_lock(lock)


def test_lock_validation_rejects_bad_mel():
    base = make_lock()
    lock = make_lock(mel=replace(base.mel, hop_length=0))
    with pytest.raises(R.ReferenceError):
        R.validate_lock(lock)


def test_lock_validation_rejects_negative_contract():
    base = make_lock()
    contract = replace(base.numerical_contract or R.NumericalContract(
        encoder_max_abs_err=0.0, encoder_mean_abs_err=0.0,
        encoder_rel_l2_err=0.0, joint_token_logit_max_abs_err=0.0,
        joint_duration_logit_max_abs_err=0.0,
        token_ids_must_match_exactly=True, transcript_must_match_exactly=True,
        nan_count_allowed=0, inf_count_allowed=0, derived_from="t",
    ), encoder_max_abs_err=-1.0)
    lock = make_lock(numerical_contract=contract)
    with pytest.raises(R.ReferenceError):
        R.validate_lock(lock)


def test_lock_validation_rejects_missing_package():
    lock = make_lock(files=[R.LockedFile(path="tokenizer.json", size=2, sha256="d" * 64)])
    with pytest.raises(R.ReferenceError, match="encoder.mlpackage"):
        R.validate_lock(lock)


def test_lock_validation_rejects_duplicate_paths():
    lock = make_lock()
    lock.files.append(R.LockedFile(path=lock.files[0].path, size=2, sha256="d" * 64))
    with pytest.raises(R.ReferenceError, match="duplicate"):
        R.validate_lock(lock)


def test_lock_validation_rejects_bad_file_hash():
    lock = make_lock()
    lock.files.append(R.LockedFile(path="extra.bin", size=1, sha256="z" * 64))
    with pytest.raises(R.ReferenceError, match="extra.bin"):
        R.validate_lock(lock)


# ---------- cache verification ----------


def test_verify_ok(tmp_path):
    lock, cache = small_cache(tmp_path)
    ok, mismatches = R.verify_cache(cache, lock)
    assert ok, mismatches
    assert not (cache / R.CACHE_STAMP_NAME).exists()
    R.record_stamps(cache, lock)
    assert (cache / R.CACHE_HASHES_NAME).read_text().startswith(lock.files[0].sha256)


def test_verify_missing_file(tmp_path):
    lock, cache = small_cache(tmp_path)
    (cache / "tokenizer.json").unlink()
    ok, mismatches = R.verify_cache(cache, lock)
    assert not ok
    assert any("missing: tokenizer.json" in m for m in mismatches)


def test_verify_size_mismatch(tmp_path):
    lock, cache = small_cache(tmp_path)
    (cache / "tokenizer.json").write_bytes(b'{"hello": "tokenizer"} ')
    ok, mismatches = R.verify_cache(cache, lock)
    assert not ok
    assert any("size-mismatch: tokenizer.json" in m for m in mismatches)


def test_verify_tampered_content_same_size(tmp_path):
    lock, cache = small_cache(tmp_path)
    data = bytearray((cache / "tokenizer.json").read_bytes())
    data[5] = data[5] ^ 0xFF  # same size, flipped byte
    (cache / "tokenizer.json").write_bytes(bytes(data))
    ok, mismatches = R.verify_cache(cache, lock)
    assert not ok
    assert any("sha256-mismatch: tokenizer.json" in m for m in mismatches)


def test_verify_never_trusts_stamps(tmp_path):
    """Forged stamp files must not make tampered content verify."""
    lock, cache = small_cache(tmp_path)
    (cache / "tokenizer.json").write_bytes(b"tampered-content")
    R.record_stamps(cache, lock)
    ok, _ = R.verify_cache(cache, lock)
    assert not ok  # verification hashes content regardless of stamps


def test_git_blob_sha1_matches_git_hash_object(tmp_path):
    payload = b"sample content for blob id\n"
    f = tmp_path / "x.txt"
    f.write_bytes(payload)
    expected = subprocess.run(
        ["git", "hash-object", "--stdin"], input=payload,
        capture_output=True, check=True,
    ).stdout.decode().strip()
    assert R.git_blob_sha1(f) == expected
