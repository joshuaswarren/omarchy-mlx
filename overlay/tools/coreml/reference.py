# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Parakeet reference lock schema and cache verification.

Defines the :data:`ReferenceLock` schema, cache layout, and hash
verification for the pinned public Parakeet reference (phase 1 of
``docs/plans/2026-09-12-coreml-parakeet-ane-plan.md``).

The lock file (``parakeet-reference.lock``) lives next to this module
under ``overlay/tools/coreml/``. It pins:

* the GitHub commit of ``mweinbach/parakeet-coreml-swift`` (the public
  reference implementation, Apache-2.0),
* the Hugging Face revision of ``mweinbach1/parakeet-tdt-0.6b-v3-coreml``
  (the public reference model, CC-BY-4.0),
* sha-256 + size of every file in that revision,
* the audio fixture (URL, revision, sha-256, license),
* the mel-feature and TDT-decoder configuration,
* the frozen numerical acceptance contract,
* the macOS Core ML reference environment used to produce the accepted
  golden outputs, and sha-256s of the captured golden tensors/tokens/
  transcript (keys absent until a capture lands).

Integrity rules (fail closed):

* Every file in the cache is hashed in full on every verification.
  The sidecar stamp files (``.sha256sums``, ``.manifest-stamp``) are
  receipts of the last successful verify, never a basis to skip
  verification: nothing is trusted on first use, mtime, or presence.
* Content that does not match the pin is never used; the downloader
  re-fetches it, and hard-fails if freshly fetched content still
  mismatches (upstream moved or systematic corruption).

Cache layout::

    $MLX_OMARCHY_CACHE_DIR/parakeet-reference/   (default on Linux:
        <model-repo>/<hf-revision>/...            ~/.cache/mlx-omarchy)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")  # git commit / HF revision sha1
CACHE_STAMP_NAME = ".manifest-stamp"
CACHE_HASHES_NAME = ".sha256sums"
CACHE_VERIFIED_NAME = ".verified-hashes"
LOCK_NAME = "parakeet-reference.lock"


_HF_BASE = "https://huggingface.co"


class ReferenceError(RuntimeError):
    """Raised for any reference-lock or cache-verification problem."""


def hf_base() -> str:
    """HF endpoint; ``MLX_OMARCHY_HF_ENDPOINT`` overrides (test/mirror)."""
    return os.environ.get("MLX_OMARCHY_HF_ENDPOINT", _HF_BASE)


def model_url(model_repo: str, revision: str, path: str) -> str:
    """Canonical resolve URL for one file of the pinned revision."""
    return f"{hf_base()}/{model_repo}/resolve/{revision}/{path}"


def tree_url(model_repo: str, revision: str) -> str:
    """HF API tree listing for the pinned revision (LFS oids = sha256)."""
    return f"{hf_base()}/api/models/{model_repo}/tree/{revision}?recursive=true"


def default_cache_root() -> Path:
    """Cache root: ``$MLX_OMARCHY_CACHE_DIR`` > XDG > ``~/.cache``."""
    base = os.environ.get("MLX_OMARCHY_CACHE_DIR")
    if base:
        return Path(base) / "parakeet-reference"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "mlx-omarchy" / "parakeet-reference"
    return Path.home() / ".cache" / "mlx-omarchy" / "parakeet-reference"


def model_cache_dir(cache_root: Path, model_repo: str, revision: str) -> Path:
    """Predictable cache directory for one Hugging Face model revision."""
    return cache_root / model_repo / revision


def default_lock_path() -> Path:
    return Path(__file__).resolve().parent / LOCK_NAME


# ---------- Lock schema ----------


@dataclass
class LockedFile:
    """One pinned file inside the reference model revision."""

    path: str    # repo-relative path, forward slashes
    size: int    # exact byte size
    sha256: str  # lowercase hex (HF LFS oid for LFS files)


@dataclass
class MelConfig:
    """Mel feature extractor configuration (reference conventions)."""

    sample_rate: int     # 16000
    hop_length: int      # 160
    win_length: int      # 400 (symmetric/non-periodic Hann)
    n_fft: int           # 512, centred frames, zero pad-mode
    n_mels: int          # 128, Slaney-normalised, f_min 0, f_max sr/2
    preemphasis: float   # 0.97, y[0] = x[0]
    log_guard: float     # 2^-24, added before log
    epsilon: float       # 1e-5, denominator of per-bin normalisation


@dataclass
class TdtConfig:
    """Greedy TDT decoder configuration (reference conventions)."""

    blank_token_id: int        # 8192
    durations: list[int]       # [0, 1, 2, 3, 4]
    max_symbols_per_step: int  # 10
    vocab_size: int            # 8193


@dataclass
class AudioFixture:
    """Pinned speech input; the lock records its identity and license."""

    url: str                 # exact-revision source URL
    sha256: str
    size: int
    sample_rate: int
    duration_seconds: float
    license: str
    note: str


@dataclass
class MacosReferenceEnvironment:
    """macOS environment that produced the accepted golden outputs."""

    chip: str            # e.g. "Apple M1 Ultra" (never a different SoC label)
    macos_release: str   # e.g. "macOS 26.6.2 (25G83)"
    coremltools_version: str   # converter toolchain recorded in the package spec
    coreml_framework_version: str | None  # runtime framework build, from capture
    swift_toolchain: str
    compute_units: str   # e.g. "cpuAndNeuralEngine" (ParakeetComputeUnits.ane)
    parakeet_coreml_swift_commit: str


@dataclass
class NumericalContract:
    """Frozen acceptance thresholds for the Linux-vs-golden comparison.

    Derived from measured divergence between compute plans on the
    reference Mac (see docs/parakeet.md); not loosened after failures.
    """

    encoder_max_abs_err: float
    encoder_mean_abs_err: float
    encoder_rel_l2_err: float
    # Joint logits are internal to the reference decode loop; the enforced
    # criterion is exact token/duration/frame sequence equality
    # (token_ids_must_match_exactly). The bounds are None = not directly
    # compared.
    joint_token_logit_max_abs_err: float | None
    joint_duration_logit_max_abs_err: float | None
    token_ids_must_match_exactly: bool
    transcript_must_match_exactly: bool
    nan_count_allowed: int
    inf_count_allowed: int
    derived_from: str


@dataclass
class ReferenceLock:
    """Parsed :file:`parakeet-reference.lock`."""

    schema_version: int
    reference_repo: str
    reference_commit: str
    model_repo: str
    model_revision: str
    model_license: str
    model_quantization: str
    files: list[LockedFile]
    audio: AudioFixture
    mel: MelConfig
    tdt: TdtConfig
    numerical_contract: NumericalContract | None  # None until frozen
    macos_reference_environment: MacosReferenceEnvironment | None
    # macos_reference_paths maps logical names to sha-256 of the golden
    # artifact: waveform, mel, mel_mask, encoder_input_features,
    # encoder_input_mask, encoder_hidden, encoder_mask, token_ids,
    # transcript, capture_log, manifest. Keys are absent until captured.
    macos_reference_paths: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> ReferenceLock:
        if not isinstance(d, dict):
            raise ReferenceError("reference lock must be a JSON object")
        try:
            contract = d.get("numerical_contract")
            env = d.get("macos_reference_environment")
            return cls(
                schema_version=int(d["schema_version"]),
                reference_repo=str(d["reference_repo"]),
                reference_commit=str(d["reference_commit"]),
                model_repo=str(d["model_repo"]),
                model_revision=str(d["model_revision"]),
                model_license=str(d["model_license"]),
                model_quantization=str(d["model_quantization"]),
                files=[LockedFile(**f) for f in d["files"]],
                audio=AudioFixture(**d["audio"]),
                mel=MelConfig(**d["mel"]),
                tdt=TdtConfig(**d["tdt"]),
                numerical_contract=(
                    NumericalContract(**contract) if contract else None
                ),
                macos_reference_environment=(
                    MacosReferenceEnvironment(**env) if env else None
                ),
                macos_reference_paths=dict(d.get("macos_reference_paths", {})),
            )
        except (KeyError, TypeError) as ex:
            raise ReferenceError(f"invalid reference lock: {ex}") from ex

    def to_dict(self) -> dict:
        def _or_none(x):
            return asdict(x) if x is not None else None

        return {
            "schema_version": self.schema_version,
            "reference_repo": self.reference_repo,
            "reference_commit": self.reference_commit,
            "model_repo": self.model_repo,
            "model_revision": self.model_revision,
            "model_license": self.model_license,
            "model_quantization": self.model_quantization,
            "files": [asdict(f) for f in self.files],
            "audio": asdict(self.audio),
            "mel": asdict(self.mel),
            "tdt": asdict(self.tdt),
            "numerical_contract": _or_none(self.numerical_contract),
            "macos_reference_environment": _or_none(
                self.macos_reference_environment
            ),
            "macos_reference_paths": dict(self.macos_reference_paths),
        }

    @classmethod
    def load(cls, path: Path | None = None) -> ReferenceLock:
        path = path or default_lock_path()
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as ex:
            raise ReferenceError(f"reference lock not found: {path}") from ex
        try:
            data = json.loads(text)
        except json.JSONDecodeError as ex:
            raise ReferenceError(f"reference lock is not valid JSON: {ex}") from ex
        return cls.from_dict(data)


def write_lock(lock: ReferenceLock, path: Path) -> None:
    """Write the lock in sorted, stable JSON form for diff review."""
    payload = lock.to_dict()
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_lock(lock: ReferenceLock) -> None:
    """Fail closed on a malformed lock. No I/O."""
    if lock.schema_version != 1:
        raise ReferenceError(
            f"unsupported reference lock schema_version={lock.schema_version}"
        )
    for name in ("reference_commit", "model_revision"):
        value = getattr(lock, name)
        if not _REVISION_RE.match(value):
            raise ReferenceError(f"{name} must be 40-char hex, got {value!r}")
    if not lock.files:
        raise ReferenceError("reference lock contains no files")
    seen: set[str] = set()
    for f in lock.files:
        if f.path in seen:
            raise ReferenceError(f"duplicate file entry: {f.path}")
        seen.add(f.path)
        if not _SHA256_RE.match(f.sha256):
            raise ReferenceError(f"{f.path}: sha256 must be 64-char hex")
        if f.size < 0:
            raise ReferenceError(f"{f.path}: negative size")
    for name in ("encoder.mlpackage", "decoder.mlpackage", "joint.mlpackage"):
        if not any(f.path.startswith(name + "/") for f in lock.files):
            raise ReferenceError(f"lock is missing {name} package files")
    if not any(f.path == "tokenizer.json" for f in lock.files):
        raise ReferenceError("lock is missing tokenizer.json")
    if not _SHA256_RE.match(lock.audio.sha256):
        raise ReferenceError("audio fixture sha256 must be 64-char hex")
    if lock.audio.size <= 0:
        raise ReferenceError("audio fixture size must be positive")
    if lock.tdt.blank_token_id >= lock.tdt.vocab_size:
        raise ReferenceError(
            f"tdt.blank_token_id={lock.tdt.blank_token_id} >= "
            f"vocab_size={lock.tdt.vocab_size}"
        )
    if not lock.tdt.durations:
        raise ReferenceError("tdt.durations must not be empty")
    if lock.tdt.max_symbols_per_step <= 0:
        raise ReferenceError("tdt.max_symbols_per_step must be > 0")
    if lock.mel.n_fft <= 0 or lock.mel.hop_length <= 0 or lock.mel.win_length <= 0:
        raise ReferenceError("mel.n_fft/hop_length/win_length must be positive")
    if lock.mel.n_mels <= 0:
        raise ReferenceError("mel.n_mels must be positive")
    contract = lock.numerical_contract
    if contract is not None:
        for bound in (
            contract.encoder_max_abs_err,
            contract.encoder_mean_abs_err,
            contract.encoder_rel_l2_err,
            contract.joint_token_logit_max_abs_err,
            contract.joint_duration_logit_max_abs_err,
        ):
            if bound is not None and bound < 0:
                raise ReferenceError("numerical contract bounds must be >= 0")
        if contract.nan_count_allowed < 0 or contract.inf_count_allowed < 0:
            raise ReferenceError("numerical contract NaN/Inf allowance negative")

def git_blob_sha1(path: Path) -> str:
    """Git blob object id (sha1) of a file's content."""
    data = path.read_bytes()
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


# ---------- Hashing and cache verification ----------


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """FIPS 180-4 SHA-256 of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_cache_full(
    cache_dir: Path, lock: ReferenceLock
) -> tuple[bool, list[str], set[str]]:
    """Verify and return ``(ok, mismatches, bad_paths)``."""
    mismatches: list[str] = []
    bad: set[str] = set()
    for f in lock.files:
        p = cache_dir / f.path
        if not p.is_file():
            mismatches.append(f"missing: {f.path}")
            bad.add(f.path)
            continue
        try:
            actual_size = p.stat().st_size
        except OSError as ex:
            mismatches.append(f"stat-fail: {f.path}: {ex}")
            bad.add(f.path)
            continue
        if actual_size != f.size:
            mismatches.append(
                f"size-mismatch: {f.path}: expected {f.size}, got {actual_size}"
            )
            bad.add(f.path)
            continue
        actual_sha = sha256_file(p)
        if actual_sha != f.sha256:
            mismatches.append(
                f"sha256-mismatch: {f.path}: expected {f.sha256}, got {actual_sha}"
            )
            bad.add(f.path)
    return (not mismatches), mismatches, bad


def verify_cache(cache_dir: Path, lock: ReferenceLock) -> tuple[bool, list[str]]:
    """Verify every pinned file is present with matching size + sha256.

    Always hashes full file contents; never trusts presence, size, or
    prior stamps. Returns ``(ok, mismatches)`` without raising.
    """
    ok, mismatches, _ = _verify_cache_full(cache_dir, lock)
    return ok, mismatches


def trust_cache(
    cache_dir: Path, lock: ReferenceLock
) -> tuple[bool, list[str]]:
    """Validate the cache without re-reading the file contents.

    Returns ``(ok, mismatches)`` without raising. ``ok`` is True ONLY
    when every locked file is present, its size matches, its mtime is
    no newer than the ``.verified-hashes`` sidecar's mtime, and the
    sidecar's recorded hash agrees with the lock's pinned sha256. The
    sidecar's mtime is the moment the last successful verify finished;
    any cache write or partial download that bumps a file's mtime past
    that point is detected here without opening the file. The recorded
    hash sanity-check pins the sidecar to the lock, so a sidecar from
    an older revision cannot pass against the current lock.
    """
    mismatches: list[str] = []
    verified_path = cache_dir / CACHE_VERIFIED_NAME
    if not verified_path.is_file():
        mismatches.append(
            f"missing-stamp: {CACHE_VERIFIED_NAME} "
            f"(run `mlx-omarchy-parakeet verify` to refresh)")
        return False, mismatches
    try:
        sidecar_mtime_ns = verified_path.stat().st_mtime_ns
        verified_lines = verified_path.read_text(encoding="utf-8").splitlines()
    except OSError as ex:
        mismatches.append(f"stamp-read-fail: {ex}")
        return False, mismatches
    recorded = {}
    for line in verified_lines:
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            sha, _, path = line.partition("  ")
            recorded[path] = sha
        except ValueError:
            mismatches.append(f"verified-hashes-malformed: {line!r}")
    if mismatches:
        return False, mismatches
    for f in lock.files:
        p = cache_dir / f.path
        if not p.is_file():
            mismatches.append(f"missing: {f.path}")
            continue
        try:
            st = p.stat()
        except OSError as ex:
            mismatches.append(f"stat-fail: {f.path}: {ex}")
            continue
        if st.st_size != f.size:
            mismatches.append(
                f"size-mismatch: {f.path}: lock={f.size}, "
                f"actual={st.st_size}")
            continue
        if st.st_mtime_ns > sidecar_mtime_ns:
            mismatches.append(
                f"newer-than-stamp: {f.path} "
                f"(file mtime={st.st_mtime_ns}, "
                f"sidecar mtime={sidecar_mtime_ns})")
            continue
        side_sha = recorded.get(f.path)
        if side_sha is None:
            mismatches.append(f"sha-stamp-missing: {f.path}")
            continue
        if side_sha != f.sha256:
            mismatches.append(
                f"sha-stamp-mismatch: {f.path}: sidecar={side_sha}, "
                f"lock={f.sha256}")
    return (not mismatches), mismatches


def verify_cache_with_stamp(
    cache_dir: Path,
    lock: ReferenceLock,
    *,
    trust_stamp: bool,
) -> tuple[bool, list[str]]:
    """``verify_cache`` unless ``trust_stamp`` lets the sidecar win.

    When ``trust_stamp`` is True and :func:`trust_cache` validates every
    locked file from the verified-hashes sidecar's mtime and recorded
    hash, the full SHA-256 sweep is skipped. A trust failure (missing
    stamp, size mismatch, mtime drift, recorded-vs-lock hash drift) falls
    through to :func:`verify_cache` and refreshes the sidecar so a
    later stamp-only run can take it. With ``trust_stamp`` False the
    call is a straight ``verify_cache`` and stamps are ignored — same
    behavior as before this wrapper existed.
    """
    if trust_stamp:
        ok, mismatches = trust_cache(cache_dir, lock)
        if ok:
            return True, []
        ok, mismatches = verify_cache(cache_dir, lock)
        if ok:
            record_verified_hashes(cache_dir, lock)
            record_stamps(cache_dir, lock)
        return ok, mismatches
    return verify_cache(cache_dir, lock)


def record_verified_hashes(
    cache_dir: Path, lock: ReferenceLock
) -> None:
    """Hash every pinned file and write ``.verified-hashes``.

    Each line is ``<actual_sha256>  <path>``, the bytes that were on disk
    the moment this ran. ``trust_cache`` uses the sidecar's mtime to
    detect any post-verify write and the recorded hash to bind the
    sidecar to the current lock's pins.
    """
    hash_lines = []
    for f in lock.files:
        p = cache_dir / f.path
        if not p.is_file():
            continue
        hash_lines.append(f"{sha256_file(p)}  {f.path}\n")
    (cache_dir / CACHE_VERIFIED_NAME).write_text(
        "".join(hash_lines), encoding="utf-8"
    )


def record_stamps(cache_dir: Path, lock: ReferenceLock) -> None:
    """Write sidecar receipt files after a successful verify.

    ``.manifest-stamp`` and ``.sha256sums`` mirror the lock for human
    readers (and ``sha256sum -c``); ``.verified-hashes`` records the
    actual bytes SHA captured at the same moment so the opt-in stamp
    trust path on ``transcribe`` can short-circuit re-hashing on a
    follow-up run without losing tamper coverage.
    """
    stamp_lines = []
    hash_lines = []
    for f in lock.files:
        stamp_lines.append(f"{f.size}\t{f.path}\n")
        hash_lines.append(f"{f.sha256}  {f.path}\n")
    (cache_dir / CACHE_STAMP_NAME).write_text(
        "".join(stamp_lines), encoding="utf-8"
    )
    (cache_dir / CACHE_HASHES_NAME).write_text(
        "".join(hash_lines), encoding="utf-8"
    )
    record_verified_hashes(cache_dir, lock)
