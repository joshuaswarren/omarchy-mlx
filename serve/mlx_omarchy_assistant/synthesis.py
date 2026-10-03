"""Offline speech synthesis for the Omarchy assistant.

Default pinned pack, verified against primary sources (2026-09-29/30):
mlx-community/Kokoro-82M-bf16 @ a71e4d38b236d968966a2002c4c895dbd12b1c3c
(Apache-2.0, 5 files sha256-pinned in KOKORO_PACK; 24 kHz) run by
mlx-audio 0.5.6 (wheel sha256 7cf7b4913...d8b4c): the Kokoro pipeline
phonemizes with misaki plus the espeakng-loader/phonemizer user-space
wheels (no root, no system package), loads voices from the local pack,
and never touches the network at run time. It needs the mlx-omarchy
wheel 0.7.17+ (in-shader trig reduction and conv decomposition); older
wheels are refused with an upgrade message instead of a backend crash.
Qwen3-TTS (VOICE_PACK, mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit
@ 08c72cad5e2f, Apache-2.0, 12 files sha256-pinned) remains the selectable
second engine. The default engine is VOICE_ENGINES[0] (owner decision,
2026-10-02: Kokoro default, voice af_heart, no listening step); an
explicitly saved voice choice always wins over the default.

Requests run in one persistent owned worker process (model resident;
wedge/cancel resets it; piped IPC only, no sync primitives).  state is
missing/usable/ready and never implies qualification: qualified is True
only while the durable qualification.json under the assets dir ties the
running mlx binary hash and pinned model hash to a listener-verified,
zero-CPU-dispatch, latency-measured hardware receipt.
"""

from __future__ import annotations

import array
import atexit
import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import multiprocessing
import os
import re
import sys
import tempfile
import threading
import time
import urllib.request
import wave
from pathlib import Path
from typing import Iterator, NamedTuple

MAX_TEXT_CHARS = 4000
MAX_OUTPUT_SECONDS = 30.0
MAX_PENDING_REQUESTS = 2
FIRST_MESSAGE_TIMEOUT = 120.0
MESSAGE_TIMEOUT = 15.0
CANCEL_GRACE = 2.0
WORKER_STOP_TIMEOUT = 5.0
GENERATION_DEADLINE_SECONDS = 300.0
PROBE_TTL_SECONDS = 30.0
MP_START_METHOD = "spawn"
_RECEIPT_NAME = "manifest.json"
_QUALIFICATION_NAME = "qualification.json"
_URL_TIMEOUT = 120
_GRACEFUL_JOIN = 1.0

VOICE_PACK = {
    "id": "qwen3-tts-0.6b-customvoice-4bit",
    "repo": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
    "revision": "08c72cad5e2fd0f41730c8bd1f28149585e46361",
    "license": "apache-2.0",
    "voice": "aiden",
    "voices": ["serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna",
               "sohee", "eric", "dylan"],
    "asset_bytes": 1693602151,
    "weights_bytes": 1689065612,
    "runtime_estimate_bytes": 1957501068,
    "files": [
        {"name": "config.json", "bytes": 6058,
         "sha256": "612cb591b44547319e5c68a78c0e93e4defb57882a4aa9ef5f06cc2f071ed036"},
        {"name": "generation_config.json", "bytes": 245,
         "sha256": "f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa"},
        {"name": "merges.txt", "bytes": 1671839,
         "sha256": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"},
        {"name": "model.safetensors", "bytes": 1006772520,
         "sha256": "4ab02a20be381700f6e73dbb5efdc424cadf9f1d0652cbffd662872ea41e296a"},
        {"name": "model.safetensors.index.json", "bytes": 71447,
         "sha256": "f3b84ec5c1b38220008c3a300b8a73502f7d4ec67b232e0193d9376909fc4e3e"},
        {"name": "preprocessor_config.json", "bytes": 127,
         "sha256": "efdde1022ea9d76928bf7a9cd53139138f5ba2e466e837f08f6105ab1af1c119"},
        {"name": "speech_tokenizer/config.json", "bytes": 2336,
         "sha256": "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167"},
        {"name": "speech_tokenizer/configuration.json", "bytes": 76,
         "sha256": "6bc26d64eb5024b4d1dab5a52371958b429256d6c9d59787f1f5294a54e0cebd"},
        {"name": "speech_tokenizer/model.safetensors", "bytes": 682293092,
         "sha256": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"},
        {"name": "speech_tokenizer/preprocessor_config.json", "bytes": 234,
         "sha256": "fcb3805e597e786d4067706e602f6688524640f8d3396790e2e09b5942fcbdfb"},
        {"name": "tokenizer_config.json", "bytes": 7344,
         "sha256": "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670"},
        {"name": "vocab.json", "bytes": 2776833,
         "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"},
    ],
    "runtime": {
        "mlx_audio": {
            "version": "0.5.6",
            "wheel_sha256": ("7cf7b49135f6f681988a9e0b9d2fbfdd2e9be783b0a3204"
                             "d0c4028e0cf2d8b4c"),
        },
        "requires": ["mlx", "mlx_audio", "transformers", "numpy"],
        "constraints": {
            "mlx_audio": "==0.5.6",
            "transformers": ">=5.14.0",
            "numpy": ">=1.26.4",
        },
    },
}

_URL_TEMPLATE = "https://huggingface.co/{repo}/resolve/{revision}/{name}"

# Default engine (owner decision 2026-10-02): Kokoro-82M (StyleTTS2-based,
# non-autoregressive — one or a few forward passes per sentence instead of
# one per 12.5 Hz frame). Pinned like the second engine; 24 kHz; American
# English voices from the pack's own voice files.
KOKORO_PACK = {
    "id": "kokoro-82m-bf16",
    "repo": "mlx-community/Kokoro-82M-bf16",
    "revision": "a71e4d38b236d968966a2002c4c895dbd12b1c3c",
    "license": "apache-2.0",
    "label": "Kokoro 82M",
    "voice": "af_heart",
    "voices": ["af_heart", "af_bella", "am_michael"],
    "asset_bytes": 328684463,
    "weights_bytes": 328682112,
    "runtime_estimate_bytes": 585000000,
    "files": [
        {"name": "config.json", "bytes": 2351,
         "sha256": "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f"},
        {"name": "kokoro-v1_0.safetensors", "bytes": 327115152,
         "sha256": "4e9ecdf03b8b6cf906070390237feda473dc13327cb8d56a43deaa374c02acd8"},
        {"name": "voices/af_heart.safetensors", "bytes": 522320,
         "sha256": "2c1c733b0e6576c810e268d3e440c21dea4e0f0131a3ba4cfc98d7fe6136d094"},
        {"name": "voices/af_bella.safetensors", "bytes": 522320,
         "sha256": "112d310468cbb3cf23404d3d0b50ad3adf017b87bf38bf9edd15f4ad572df6a3"},
        {"name": "voices/am_michael.safetensors", "bytes": 522320,
         "sha256": "3940147ded35deba0bb52e8132f89b719298e0520258c34584358aa5a24da2ea"},
    ],
    "runtime": {
        "mlx_audio": {
            "version": "0.5.6",
            "wheel_sha256": ("7cf7b49135f6f681988a9e0b9d2fbfdd2e9be783b0a3204"
                             "d0c4028e0cf2d8b4c"),
        },
        "requires": ["mlx", "mlx_audio", "numpy", "misaki", "num2words",
                     "spacy", "en-core-web-sm", "phonemizer", "espeakng-loader"],
        "constraints": {
            "mlx_audio": "==0.5.6",
            "misaki": "==0.7.4",
            "num2words": "==0.5.14",
            "spacy": ">=3.8.0",
            "en-core-web-sm": "==3.8.0",
            "phonemizer": ">=3.2.1",
            "espeakng-loader": "==0.2.4",
        },
    },
}

VOICE_PACK["label"] = "Qwen3-TTS 0.6B CustomVoice"

# Engine registry keyed implicitly by pack id; read at call time so tests
# can patch any pack. VOICE_ENGINES[0] is the default engine everywhere an
# unset choice must resolve; the order is also the UI order (default first).
VOICE_ENGINES = (KOKORO_PACK, VOICE_PACK)

# Native language per Kokoro preset speaker: all American English, from the
# pack's own voice tensors (voices/<id>.safetensors).
KOKORO_VOICE_META = {
    "af_heart": {"label": "Heart", "accent": "American English"},
    "af_bella": {"label": "Bella", "accent": "American English"},
    "am_michael": {"label": "Michael", "accent": "American English"},
}


# Native language per preset speaker, shown as a plain label and accent note.
# The pack has no American English female voice; non-English-native voices
# reading English text are offered openly, accent included.
VOICE_META = {
    "aiden": {"label": "Aiden", "accent": "American English"},
    "ryan": {"label": "Ryan", "accent": "English"},
    "serena": {"label": "Serena", "accent": "Chinese-native; English has an accent"},
    "vivian": {"label": "Vivian", "accent": "Chinese-native; English has an accent"},
    "uncle_fu": {"label": "Uncle Fu", "accent": "Chinese-native; English has an accent"},
    "ono_anna": {"label": "Ono Anna", "accent": "Japanese-native; English has an accent"},
    "sohee": {"label": "Sohee", "accent": "Korean-native; English has an accent"},
    "eric": {"label": "Eric", "accent": "Sichuan dialect (Chinese)"},
    "dylan": {"label": "Dylan", "accent": "Beijing dialect (Chinese)"},
}
_VOICE_CHOICE_NAME = "voice.json"


def voice_options() -> list[dict]:
    """Every engine's preset speakers with label and accent, default engine
    first; each option names its engine so the UI can group honestly."""
    options = []
    for pack in VOICE_ENGINES:
        for name in pack["voices"]:
            entry = dict(_voice_meta(name))
            entry["id"] = name
            entry["engine"] = pack["id"]
            entry["engine_label"] = pack.get("label", pack["id"])
            options.append(entry)
    return options


def _voice_pack(name) -> dict | None:
    """The engine pack that declares this voice, or None."""
    for pack in VOICE_ENGINES:
        if name in pack["voices"]:
            return pack
    return None


def _pack_by_id(pack_id) -> dict | None:
    for pack in VOICE_ENGINES:
        if pack["id"] == pack_id:
            return pack
    return None


def _voice_meta(name) -> dict:
    pack = _voice_pack(name)
    if pack is not None and pack["id"] == KOKORO_PACK["id"]:
        return KOKORO_VOICE_META[name]
    return VOICE_META[name]


def resolve_voice(name) -> str:
    """Validate a requested voice across engines; unknown is a named
    refusal, never a fallback to the default."""
    if not isinstance(name, str) or _voice_pack(name) is None:
        raise VoiceError(
            f"unknown voice {name!r}; available voices: "
            + ", ".join(o["id"] for o in voice_options()))
    return name


_EXPORTED_ERRORS = {name: None for name in
                    ("VoiceError", "VoiceAssetsMissingError",
                     "VoiceDependencyMissingError", "AcceleratorUnavailableError",
                     "VoiceBusyError", "VoiceOutputLimitError")}


class VoiceError(Exception):
    """Named voice failure; message names the component and fix."""


class VoiceAssetsMissingError(VoiceError):
    pass


class VoiceDependencyMissingError(VoiceError):
    pass


class AcceleratorUnavailableError(VoiceError):
    pass


class VoiceBusyError(VoiceError):
    pass


class VoiceOutputLimitError(VoiceError):
    pass


class SynthesisCancelled(Exception):
    """Raised when the caller's cancel event fires during synthesis."""


for _name in _EXPORTED_ERRORS:
    _EXPORTED_ERRORS[_name] = globals()[_name]


class PcmChunk(NamedTuple):
    """One bounded synthesis chunk: mono PCM16 little-endian samples."""

    sample_rate: int
    data: bytes


# Verified-hash cache: in-memory, process-lifetime; a fresh process hashes the
# pack once.  Identity includes st_ctime_ns (writes bump it, os.utime cannot
# forge it back); identities are never read from the mutable receipt.
_stat_identity_cache: dict = {}
_stat_identity_lock = threading.Lock()
_live_workers: set = set()
_workers_lock = threading.Lock()


def forget_asset_cache() -> None:
    with _stat_identity_lock:
        _stat_identity_cache.clear()


def _identity(st) -> tuple:
    return (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)


def _version_satisfies(version: str, constraint: str) -> bool | None:
    m = re.fullmatch(r"(>=|==)\s*(\d+(?:\.\d+)*)", constraint.strip())
    if not m:
        return None
    current = tuple(int(part) for part in version.split(".") if part.isdigit())
    spec = tuple(int(part) for part in m.group(2).split("."))
    width = max(len(current), len(spec))
    current += (0,) * (width - len(current))
    spec += (0,) * (width - len(spec))
    return current >= spec if m.group(1) == ">=" else current == spec


def _constraint_conflicts(deps: dict) -> list[str]:
    return [f"{name} {info['version']} violates {info['constraint']}"
            for name, info in deps["detail"].items()
            if isinstance(info, dict) and info.get("satisfies") is False]


def _probe_dependencies_now(pack: dict | None = None) -> dict:
    pack = pack if pack is not None else VOICE_ENGINES[0]
    present, missing, detail = [], [], {}
    constraints = pack["runtime"]["constraints"]
    for name in pack["runtime"]["requires"]:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            try:
                module = importlib.import_module(name)
                version = getattr(module, "__version__", "unknown")
            except Exception as exc:
                missing.append(name)
                detail[name] = f"{type(exc).__name__}: {exc}"
                continue
        present.append(name)
        entry: dict = {"version": version}
        constraint = constraints.get(name)
        if constraint:
            entry["constraint"] = constraint
            entry["satisfies"] = _version_satisfies(version, constraint)
        detail[name] = entry
    return {"present": present, "missing": missing, "detail": detail}


def _probe_accelerator_now() -> dict:
    try:
        import mlx.core as mx
    except Exception as exc:
        return {"available": False, "device": None,
                "detail": f"mlx.core not importable: {type(exc).__name__}: {exc}"}
    device = str(mx.default_device())
    return {"available": "gpu" in device, "device": device, "detail": device}


_probe_cache: dict = {}
_probe_cache_lock = threading.Lock()


def reset_probe_cache() -> None:
    with _probe_cache_lock:
        _probe_cache.clear()


def _cached_probe(key: str, fn) -> dict:
    now = time.monotonic()
    with _probe_cache_lock:
        entry = _probe_cache.get(key)
        if entry and now - entry[0] < PROBE_TTL_SECONDS:
            return entry[1]
    value = fn()
    with _probe_cache_lock:
        _probe_cache[key] = (now, value)
    return value


def probe_dependencies(pack: dict | None = None) -> dict:
    return _cached_probe(f"deps:{'default' if pack is None else pack['id']}",
                         lambda: _probe_dependencies_now(pack))


def probe_accelerator() -> dict:
    return _cached_probe("accel", _probe_accelerator_now)


def read_receipt(assets_dir: Path) -> dict | None:
    try:
        receipt = json.loads((assets_dir / _RECEIPT_NAME).read_text())
    except (OSError, ValueError):
        return None
    return receipt if isinstance(receipt, dict) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=_URL_TIMEOUT) as response:
        with dest.open("wb") as fh:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                fh.write(chunk)


def to_wav_bytes(samples, sample_rate: int,
                 max_seconds: float = MAX_OUTPUT_SECONDS) -> bytes:
    rate = int(sample_rate)
    if rate <= 0:
        raise ValueError(f"invalid sample rate: {sample_rate!r}")
    cap = int(rate * max_seconds)
    data = array.array("f")
    data.extend(samples)
    if cap < len(data):
        del data[cap:]
    if not data:
        raise ValueError("no audio samples to encode")
    pcm = array.array("h", (max(-32768, min(32767, int(s * 32767.0)))
                            for s in data))
    if sys.byteorder == "big":
        pcm.byteswap()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")


def split_sentences(text: str) -> list[str]:
    """Visible-sentence split for read-aloud: drops code blocks and raw URLs."""
    lines = []
    inside_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            inside_fence = not inside_fence
            continue
        if not inside_fence:
            lines.append(_URL_PATTERN.sub("", line))
    parts = []
    blob = " ".join(" ".join(lines).split())
    for part in _SENTENCE_END.split(blob):
        part = part.strip()
        if part:
            parts.append(part)
    return parts


def _model_pin() -> dict:
    """The primary weights file: the largest safetensors in the pack."""
    return max((e for e in VOICE_ENGINES[0]["files"]
                if e["name"].endswith(".safetensors")),
               key=lambda e: e["bytes"])


def _loaded_libmlx_path(package_dir: Path) -> Path | None:
    """The libmlx.so actually mapped into this process, else the packaged one.

    The dynamic linker, not the import system, picks the library; a stray
    LD_LIBRARY_PATH or shadowing wheel tree can silently swap the GPU
    backend build under a measurement, so the mapped path wins.
    """
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                path = line.rstrip("\n").rpartition("  ")[2]
                if path.endswith("/libmlx.so"):
                    return Path(path)
    except OSError:
        pass
    packaged = package_dir / "lib" / "libmlx.so"
    return packaged if packaged.is_file() else None


def _record_hashes(dist) -> dict:
    """Map 'mlx/lib/libmlx.so' -> expected sha256 from the dist RECORD."""
    import base64

    out = {}
    for f in dist.files or []:
        if f.hash is None or f.hash.mode != "sha256":
            continue
        out[str(f)] = base64.urlsafe_b64decode(f.hash.value + "==").hex()
    return out


def _provenance_from_paths(extension_path, libmlx_path, record_map,
                           dist_version, mx_version) -> dict:
    """Pure provenance core: hash real binaries against RECORD entries.

    ``record_map`` keys are wheel-relative paths ('mlx/lib/libmlx.so').
    Verdicts: "match", "mismatch" (hash conflict, missing backend binary,
    or version disagreement), "unverified" (no RECORD entries to check).
    """
    files = []
    identity = {}
    details = []
    record_checks = 0
    record_failures = 0
    for label, relative, path in (
        ("mlx.core extension",
         f"mlx/{Path(extension_path).name}" if extension_path else None,
         extension_path),
        ("mlx/lib/libmlx.so", "mlx/lib/libmlx.so", libmlx_path),
    ):
        if path is None or not Path(path).is_file():
            files.append({"label": label, "path": str(path),
                          "present": False})
            details.append(f"{label} is not present on disk")
            continue
        actual = _sha256_file(Path(path))
        expected = record_map.get(relative)
        match = None if expected is None else actual == expected
        files.append({"label": label, "path": str(path), "present": True,
                      "sha256": actual, "record_sha256": expected,
                      "match": match})
        identity["extension_sha256" if label.startswith("mlx.core")
                 else "libmlx_sha256"] = actual
        if expected is None:
            details.append(f"{label} carries no RECORD hash entry")
            continue
        record_checks += 1
        if not match:
            record_failures += 1
            details.append(
                f"{label} at {path}: on disk sha256 {actual} but the "
                f"installed wheel's RECORD says {expected}; the runtime "
                f"binary is not the wheel this environment claims to have "
                f"installed")
    version_conflict = (
        dist_version is not None and mx_version is not None
        and dist_version != mx_version)
    if version_conflict:
        details.append(
            f"compiled mx.__version__ {mx_version!r} != installed "
            f"distribution version {dist_version!r}; the loaded library is "
            f"not the installed wheel")
    if any(not f["present"] for f in files) or record_failures \
            or version_conflict:
        verified = "mismatch"
    elif record_checks == 0:
        verified = "unverified"
    else:
        verified = "match"
    return {
        "present": bool(files) and all(f["present"] for f in files),
        "verified": verified,
        "detail": "; ".join(details) or None,
        "identity": identity if verified == "match" else None,
        "files": files,
        "dist_version": dist_version,
        "mx_version": mx_version,
    }


def _backend_provenance() -> dict:
    """Provenance of the loaded mlx backend: extension + linked libmlx.so.

    Qualification must tie to the actual backend binary, not the Python
    extension alone; a swapped libmlx.so invalidates it.
    """
    try:
        import mlx.core as mx
    except Exception as exc:
        return {"present": False, "verified": "no-mlx",
                "detail": f"mlx.core not importable: {exc}", "identity": None,
                "files": [], "dist_version": None, "mx_version": None}
    mx_version = getattr(mx, "__version__", None)
    try:
        dist = importlib.metadata.distribution("mlx-omarchy")
    except importlib.metadata.PackageNotFoundError:
        return {"present": False, "verified": "no-metadata",
                "detail": "no mlx-omarchy distribution metadata; mlx appears "
                          "to be a source install, so installed-binary "
                          "provenance cannot be checked",
                "identity": None, "files": [],
                "dist_version": None, "mx_version": mx_version}
    package_dir = Path(dist.locate_file("mlx"))
    origin = getattr(getattr(mx, "__spec__", None), "origin", None)
    extension_path = Path(origin) if origin else None
    libmlx_path = _loaded_libmlx_path(package_dir)
    result = _provenance_from_paths(
        extension_path, libmlx_path, _record_hashes(dist),
        dist.version, mx_version,
    )
    result["libmlx_loaded_path"] = (
        str(libmlx_path) if libmlx_path else None)
    return result


def _worker_guard(assets_dir: str, pack: dict | None = None) -> None:
    pack = pack if pack is not None else VOICE_ENGINES[0]
    deps = probe_dependencies(pack)
    conflicts = _constraint_conflicts(deps)
    if deps["missing"] or conflicts:
        raise VoiceDependencyMissingError(
            "voice dependencies unsatisfied: "
            + "; ".join([", ".join(deps["missing"])] + conflicts)
            + "; install the runtime listed in status()")
    accel = probe_accelerator()
    if not accel["available"]:
        raise AcceleratorUnavailableError(
            f"voice synthesis requires the GPU accelerator "
            f"(default device {accel['device']!r}: {accel['detail']}); "
            "no CPU fallback is permitted")
    if not Synthesis._asset_status_for(Path(assets_dir), pack)["verified"]:
        raise VoiceAssetsMissingError(
            f"voice pack missing or unverified at {assets_dir}; "
            "run setup with download approval")


def _gumbel_categorical(logits, temp):
    """Drop-in for mlx_audio's ``categorical_sampling``: the same
    distribution by the Gumbel-max identity, argmax(logits/T + G) with
    G = -log(-log(U)).  ``mx.random.categorical`` takes 2.3 ms per draw on
    the M2 Vulkan backend against 0.3 ms here, and a Qwen3-TTS frame draws
    16 times (receipt 2026-09-30-speech-output-speed)."""
    import mlx.core as mx

    scaled = logits.astype(mx.float32) * (1.0 / temp)
    uniform = mx.random.uniform(shape=scaled.shape)
    return mx.argmax(scaled - mx.log(-mx.log(uniform)), axis=-1)


@contextlib.contextmanager
def _fast_codec_sampler(model):
    """Route the model module's final categorical draw through
    ``_gumbel_categorical`` for the duration of one request.  Masking,
    repetition penalty, temperature and top-k stay upstream's."""
    module = sys.modules[type(model).__module__]
    original = module.categorical_sampling
    module.categorical_sampling = _gumbel_categorical
    try:
        yield
    finally:
        module.categorical_sampling = original


_TRIG_TWO_PI = 6.2831853071795864769
# 6.28125 is exact in float32 and k*C1 stays exact for |k| < 2**17
# (observed |k| ~ 2e4); the residue is carried by C2.
_TRIG_C1 = 6.28125
_TRIG_C2 = 0.001935307179586477  # TWO_PI - C1, rounded to float32


_TRIG_PROBE_ARG = 2.0e5  # Kokoro feeds up to ~1.8e5; observed refusal at 1.27e5


def _kokoro_backend_gate(mx_module=None) -> None:
    """Refuse wheels without the in-shader trig reduction with a named
    upgrade error, instead of a mid-synthesis backend crash.

    Functional, not version-parsed: the wheel's dist version line is the
    upstream mlx lineage (0.32.4.dev...), not the release version, so no
    string compare can identify the fix. The omarchy backend refuses raw
    large-argument sin at its accuracy gate until 0.7.17 added in-shader
    reduction; one scalar probe tells the two apart. Runs before the
    graph-level reduction is installed, so it exercises the real primitive.
    """
    if mx_module is None:
        try:
            import mlx.core as mx_module  # noqa: F811
        except Exception as exc:
            raise AcceleratorUnavailableError(
                f"mlx is not importable in the voice worker: {exc}") from exc
    try:
        float(mx_module.sin(mx_module.array(_TRIG_PROBE_ARG,
                                            dtype=mx_module.float32)))
    except Exception as exc:
        if "accuracy limit" in str(exc):
            raise VoiceDependencyMissingError(
                "kokoro-82m needs the mlx-omarchy wheel 0.7.17+ (in-shader "
                "trig reduction and conv decomposition); this backend "
                "refuses large trig arguments at the accuracy gate — "
                "upgrade the wheel and retry") from exc
        raise


def _kokoro_backend_refusal(exc: Exception) -> Exception:
    """Translate the pre-0.7.17 wheel's trig-gate refusal into the named
    upgrade error, instead of a raw backend crash message."""
    if "accuracy limit" in str(exc):
        return VoiceError(
            "the mlx-omarchy wheel lacks the Kokoro trig/conv fixes "
            "(0.7.17+); upgrade the wheel and retry "
            f"(backend refusal: {exc})")
    return exc


def _kokoro_install_trig_reduction(threshold: float = 10000.0,
                                   record: list | None = None) -> None:
    """Fold sin/cos arguments into range before the primitive.

    The Kokoro graph (Snake activation sin(alpha*x), source-phase
    integration) feeds arguments past the omarchy Vulkan backend's
    100,000 float-trig accuracy gate: the driver's range reduction is
    untrusted above it and the software Payne-Hanek fallback miscompiles
    on this driver, so the backend refuses instead of computing wrong
    values. The fix is at the graph level: a 3-term Cody-Waite reduction
    in float32 (phase error ~3e-6 rad at |x| = 1e5), applied in-graph
    with mx.where so arguments under the threshold are untouched. Every
    op stays on the GPU; nothing falls back to CPU. Per-process and
    idempotent; the Qwen3-TTS engine's worker never installs it.
    """
    import mlx.core as mx
    if getattr(mx, "_kokoro_trig_reduced", False):
        return

    def reduced_arg(x):
        xf = x.astype(mx.float32)
        k = mx.floor(xf * (1.0 / _TRIG_TWO_PI))
        r = (xf - k * _TRIG_C1) - k * _TRIG_C2
        return r.astype(x.dtype)

    def wrap(real, name):
        def trig(x, *args, **kwargs):
            if record is not None:
                m = float(mx.max(mx.abs(x.astype(mx.float32))))
                record.append({"op": name, "max_arg": m,
                               "reduced": m > threshold})
            if getattr(x, "dtype", None) == mx.complex64:
                return real(x, *args, **kwargs)
            xf = x.astype(mx.float32)
            return real(mx.where(mx.abs(xf) <= threshold, x,
                                 reduced_arg(x)), *args, **kwargs)
        return trig

    real_sin, real_cos = mx.sin, mx.cos
    mx.sin = wrap(real_sin, "sin")
    mx.cos = wrap(real_cos, "cos")
    mx._kokoro_trig_reduced = True


def _kokoro_runtime(assets_dir: str):
    """Offline Kokoro pipeline over the pinned pack.

    The espeak-ng user-space wheel is registered before misaki is imported
    (mlx_audio's KokoroPipeline imports misaki.espeak at construction), so
    out-of-vocabulary words get real G2P with no root and no system package.
    Voice tensors are pre-seeded from the local pack, so no Hugging Face
    lookup ever happens at run time. The backend gate runs first: one raw
    scalar probe refuses pre-0.7.17 wheels before anything touches the
    graph, then the graph-level reduction installs.
    """
    _kokoro_backend_gate()
    _kokoro_install_trig_reduction()
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except Exception as exc:
        raise VoiceDependencyMissingError(
            "kokoro G2P runtime missing: install the espeakng-loader and "
            f"phonemizer wheels listed in status(): {exc}") from exc
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    from mlx_audio.tts.utils import load_model
    model = load_model(Path(assets_dir))
    from mlx_audio.tts.models.kokoro.pipeline import KokoroPipeline
    from mlx_audio.tts.models.kokoro.voice import load_voice_tensor
    pipe = KokoroPipeline(lang_code="a", model=model, repo_id=str(assets_dir))
    for name in KOKORO_PACK["voices"]:
        path = Path(assets_dir) / "voices" / f"{name}.safetensors"
        pipe.voices[name] = load_voice_tensor(str(path))
    return pipe


# Time-to-first-audio budget for the first model pass, in rough phonemes.
# Calibrated on the M2 (2026-10-02): misaki emits ~1.9 phonemes per unit of
# len(word)*0.6 (49 real for an est of 26 on the gate sentence), and
# KokoroPipeline.infer costs ~1.1 s per call + ~24 ms per real phoneme, so
# est 12 (~23 real phonemes) puts the first infer near ~1.65 s. The floor
# (~1.1 s per call) means the 1.5 s design target needs either a 2-3 word
# first segment or the infer floor itself shrinking (vocoder output
# streaming). The remainder streams in order right behind it.
_FIRST_SEGMENT_BUDGET = 12


def _phoneme_estimate(text: str) -> int:
    return sum(max(1, round(len(word) * 0.6)) for word in text.split())


def _first_segment(text: str,
                   budget: int = _FIRST_SEGMENT_BUDGET) -> tuple[str, str]:
    """Split read-aloud text so the first synthesis pass is short.

    Prefers the first clause boundary when it already fits the budget;
    otherwise cuts at the word boundary that keeps the estimate under it.
    Returns (first, rest); rest is empty when no cut is needed.
    """
    text = text.strip()
    if not text:
        return "", ""
    if _phoneme_estimate(text) <= budget:
        return text, ""
    clause = _CLAUSE_SPLIT.split(text, maxsplit=1)
    if len(clause) == 2 and clause[0].strip() \
            and _phoneme_estimate(clause[0]) <= budget:
        return clause[0].strip(), text[len(clause[0]):].lstrip(" \t,;:")
    taken: list[str] = []
    spent = 0
    for word in text.split():
        cost = max(1, round(len(word) * 0.6))
        if taken and spent + cost > budget:
            break
        taken.append(word)
        spent += cost
    first = " ".join(taken)
    return first, text[len(first):].lstrip()


def _kokoro_generate(pipe, text: str, voice: str) -> Iterator:
    """Yield float32 numpy chunks, one per phoneme chunk the pipeline makes.

    The first segment is bounded to a small phoneme budget so the first
    model pass is short (TTFA), then the remainder follows in order.
    """
    import numpy as np
    first, rest = _first_segment(text)
    for piece in ((first, rest) if rest else (first,)):
        for result in pipe(piece, voice=voice, speed=1.0):
            audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
            if audio.size:
                yield audio


def _worker_main(conn, assets_dir: str) -> None:
    """Persistent voice worker: model loads once, requests serialize.

    The parent's first message names the engine pack ("init"); each engine
    keeps its own resident model for the worker's lifetime.
    """
    pack: dict | None = None
    model = None
    kokoro = None
    rate = 24000
    current_id = None
    cancelled = False
    try:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                break
            kind = msg.get("type")
            if kind == "shutdown":
                break
            if kind == "init":
                requested = _pack_by_id(msg.get("pack"))
                pack = requested if requested is not None else VOICE_ENGINES[0]
                continue
            if kind == "cancel":
                if msg.get("id") == current_id:
                    cancelled = True
                continue
            if kind != "speak":
                continue
            req_id = msg["id"]
            current_id = req_id
            cancelled = False
            try:
                if pack is None:
                    pack = VOICE_ENGINES[0]
                _worker_guard(assets_dir, pack)
                if pack["id"] == KOKORO_PACK["id"]:
                    if kokoro is None:
                        kokoro = _kokoro_runtime(assets_dir)
                        rate = int(getattr(kokoro.model, "sample_rate", 24000))
                        conn.send({"type": "loaded", "sample_rate": rate})
                    import numpy as np
                    voice = resolve_voice(msg.get("voice", pack["voice"]))
                    cap = int(rate * MAX_OUTPUT_SECONDS)
                    produced = 0
                    for chunk in _kokoro_generate(kokoro, msg["text"].strip(),
                                                  voice):
                        if cancelled:
                            break
                        room = cap - produced
                        if room <= 0:
                            raise VoiceOutputLimitError(
                                "synthesis exceeded the "
                                f"{MAX_OUTPUT_SECONDS:.0f}s output bound; "
                                "split the text into shorter sentences")
                        if len(chunk) > room:
                            chunk = chunk[:room]
                        produced += len(chunk)
                        pcm = (chunk * 32767.0).clip(-32768, 32767
                                                     ).astype("int16")
                        conn.send({"type": "chunk", "id": req_id,
                                   "sample_rate": rate,
                                   "data": pcm.tobytes()})
                    conn.send({"type": "cancelled" if cancelled else "done",
                               "id": req_id})
                else:
                    if model is None:
                        from mlx_audio.tts.utils import load_model
                        model = load_model(Path(assets_dir))
                        rate = int(getattr(model, "sample_rate", 24000))
                        conn.send({"type": "loaded", "sample_rate": rate})
                    import numpy as np
                    voice = resolve_voice(msg.get("voice", pack["voice"]))
                    language = "english" if msg["text"].strip().isascii() else "auto"
                    cap = int(rate * MAX_OUTPUT_SECONDS)
                    produced = 0
                    with _fast_codec_sampler(model):
                        for result in model.generate_custom_voice(
                                text=msg["text"].strip(), speaker=voice,
                                language=language, stream=True,
                                streaming_interval=0.32):
                            if cancelled:
                                break
                            chunk = np.asarray(result.audio,
                                               dtype=np.float32).reshape(-1)
                            room = cap - produced
                            if room <= 0:
                                raise VoiceOutputLimitError(
                                    "synthesis exceeded the "
                                    f"{MAX_OUTPUT_SECONDS:.0f}s output "
                                    "bound; split the text into shorter "
                                    "sentences")
                            if len(chunk) > room:
                                chunk = chunk[:room]
                            produced += len(chunk)
                            pcm = (chunk * 32767.0).clip(
                                -32768, 32767).astype("int16")
                            conn.send({"type": "chunk", "id": req_id,
                                       "sample_rate": rate,
                                       "data": pcm.tobytes()})
                    conn.send({"type": "cancelled" if cancelled else "done",
                               "id": req_id})
            except Exception as exc:
                if pack is not None and pack["id"] == KOKORO_PACK["id"]:
                    exc = _kokoro_backend_refusal(exc)
                try:
                    conn.send({"type": "error", "id": req_id,
                               "error_type": type(exc).__name__,
                               "message": str(exc)})
                except Exception:
                    break
            finally:
                current_id = None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _stop_process(child) -> bool:
    if child.is_alive():
        child.terminate()
        child.join(WORKER_STOP_TIMEOUT)
    if child.is_alive():
        child.kill()
        child.join(WORKER_STOP_TIMEOUT)
    return not child.is_alive() and child.exitcode is not None


def _stop_all_workers() -> None:
    with _workers_lock:
        pending = list(_live_workers)
    for worker in pending:
        worker.stop()


class _WorkerHandle:
    """Owned worker process + pipe with confirmed-death teardown."""

    def __init__(self, assets_dir: str, pack_id: str):
        ctx = multiprocessing.get_context(MP_START_METHOD)
        parent_conn, worker_conn = ctx.Pipe()  # duplex: both ends send+recv
        self.conn = parent_conn
        self.pack_id = pack_id
        self.process = ctx.Process(target=_worker_main,
                                   args=(worker_conn, assets_dir),
                                   daemon=True)
        self.process.start()
        worker_conn.close()
        try:
            self.conn.send({"type": "init", "pack": pack_id})
        except Exception:
            pass
        with _workers_lock:
            _live_workers.add(self)

    def stop(self) -> bool:
        if self.process.is_alive():
            try:
                self.conn.send({"type": "shutdown"})
            except Exception:
                pass
            self.process.join(_GRACEFUL_JOIN)
        dead = (_stop_process(self.process) if self.process.is_alive()
                else self.process.exitcode is not None)
        try:
            self.conn.close()
        except Exception:
            pass
        with _workers_lock:
            _live_workers.discard(self)
        return dead


atexit.register(_stop_all_workers)


class Synthesis:
    """Local-only, accelerator-only TTS over the pinned engine packs
    (default Kokoro-82M with voice af_heart; Qwen3-TTS CustomVoice as the
    selectable second engine)."""

    _generated_once = False

    def __init__(self, home: Path):
        self.home = Path(home)
        self._worker: _WorkerHandle | None = None
        self._workers: dict[str, _WorkerHandle] = {}
        self._worker_lock = threading.Lock()
        self._slots = 0
        self._slots_cond = threading.Condition()
        self._req_counter = 0
        self._sample_rate = None
        self._last_error: str | None = None
        self._prime_done = False

    def assets_dir(self) -> Path:
        return self.home / "voice" / VOICE_ENGINES[0]["id"]

    def assets_dir_for(self, pack_id: str) -> Path:
        return self.home / "voice" / pack_id

    def _current_engine(self) -> dict:
        """The pack that owns the persisted voice choice."""
        return _voice_pack(self.current_voice()) or VOICE_ENGINES[0]

    def _voice_choice_path(self) -> Path:
        return self.home / "voice" / _VOICE_CHOICE_NAME

    def current_voice(self) -> str:
        """The voice applied to the next synthesis request. Falls back to
        the built-in default when nothing is persisted; an unreadable or
        unknown stored choice is a named refusal, never a silent fallback."""
        path = self._voice_choice_path()
        if not path.is_file():
            return VOICE_ENGINES[0]["voice"]
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise VoiceError(
                f"saved voice choice at {path} is unreadable ({exc}); "
                "pick a voice in settings to replace it") from exc
        if not isinstance(raw, dict) or "voice" not in raw:
            raise VoiceError(
                f"saved voice choice at {path} has no 'voice' field; "
                "pick a voice in settings to replace it")
        return resolve_voice(raw["voice"])

    def set_voice(self, name) -> dict:
        """Persist the next-synthesis voice choice. Atomic, mode 0600, no
        secrets. The chat workers stay alive: they pick up the new voice
        when the next request asks for it, with no reload."""
        resolved = resolve_voice(name)
        target = self._voice_choice_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp",
                                    prefix=".voice-")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump({"voice": resolved, "set_at": time.time()}, fh,
                          indent=2, sort_keys=True)
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        meta = _voice_meta(resolved)
        return {"voice": resolved, "label": meta["label"],
                "accent": meta["accent"],
                "engine": _voice_pack(resolved)["id"],
                "default": resolved == _voice_pack(resolved)["voice"]}

    @staticmethod
    def _asset_status_for(assets: Path, pack: dict | None = None) -> dict:
        pack = pack if pack is not None else VOICE_ENGINES[0]
        expected = pack["asset_bytes"]
        if not assets.is_dir():
            return {"path": str(assets), "present": False, "verified": False,
                    "bytes": 0, "expected_bytes": expected}
        receipt = read_receipt(assets)
        pinned = {entry["name"]: entry for entry in pack["files"]}
        if receipt is None or receipt.get("revision") != pack["revision"]:
            return {"path": str(assets), "present": False, "verified": False,
                    "bytes": 0, "expected_bytes": expected}
        present_bytes = 0
        verified = True
        for name, entry in pinned.items():
            path = assets / name
            if not path.is_file():
                verified = False
                break
            st = path.stat()
            if st.st_size != entry["bytes"]:
                verified = False
                break
            key = (str(assets), name)
            with _stat_identity_lock:
                cached = _stat_identity_cache.get(key)
            if cached != _identity(st):
                if _sha256_file(path) != entry["sha256"]:
                    verified = False
                    break
                with _stat_identity_lock:
                    _stat_identity_cache[key] = _identity(st)
            present_bytes += st.st_size
        if not verified:
            present_bytes = 0
        return {"path": str(assets), "present": present_bytes > 0,
                "verified": verified and present_bytes == expected,
                "bytes": present_bytes, "expected_bytes": expected}

    def _asset_status(self) -> dict:
        return self._asset_status_for(self.assets_dir())

    def status(self) -> dict:
        deps = probe_dependencies()
        accel = probe_accelerator()
        assets = self._asset_status()
        conflicts = _constraint_conflicts(deps)
        if not assets["verified"] or deps["missing"] or conflicts \
                or not accel["available"]:
            state = "missing"
        else:
            state = "ready" if self._generated_once else "usable"
        reasons = []
        if not assets["verified"]:
            reasons.append("voice pack not downloaded and hash-verified")
        if not accel["available"]:
            reasons.append(f"accelerator unavailable: {accel['detail']}")
        if deps["missing"]:
            reasons.append("missing dependencies: " + ", ".join(deps["missing"]))
        reasons.extend(conflicts)
        if state == "usable":
            reasons.append("no completed synthesis run on this machine yet")
        worker_alive = (self._worker is not None
                        and self._worker.process.is_alive())
        try:
            current = self.current_voice()
            voice_error = None
        except VoiceError as exc:
            current = VOICE_ENGINES[0]["voice"]
            voice_error = str(exc)
        engines = []
        for pack in VOICE_ENGINES:
            engine_deps = probe_dependencies(pack)
            engine_conflicts = _constraint_conflicts(engine_deps)
            engine_assets = self._asset_status_for(
                self.assets_dir_for(pack["id"]), pack)
            engines.append({
                "id": pack["id"],
                "label": pack.get("label", pack["id"]),
                "repo": pack["repo"],
                "revision": pack["revision"],
                "license": pack["license"],
                "voice_default": pack["voice"],
                "voices": list(pack["voices"]),
                "voice_options": [o for o in voice_options()
                                  if o["engine"] == pack["id"]],
                "assets": engine_assets,
                "dependencies": engine_deps,
                "usable": (engine_assets["verified"]
                           and not engine_deps["missing"]
                           and not engine_conflicts
                           and accel["available"]),
            })
        return {
            "pack": {
                "id": VOICE_ENGINES[0]["id"],
                "repo": VOICE_ENGINES[0]["repo"],
                "revision": VOICE_ENGINES[0]["revision"],
                "license": VOICE_ENGINES[0]["license"],
                "label": VOICE_ENGINES[0].get("label", VOICE_ENGINES[0]["id"]),
                "voice": current,
                "engine": (_voice_pack(current) or VOICE_ENGINES[0])["id"],
                "voice_default": VOICE_ENGINES[0]["voice"],
                "voice_options": voice_options(),
                "voices": list(VOICE_ENGINES[0]["voices"]),
                "sample_rate": self._sample_rate or 24000,
                "runtime": VOICE_ENGINES[0]["runtime"],
            },
            "engines": engines,
            "assets": assets,
            "accelerator": accel,
            "dependencies": deps,
            "memory": {
                "asset_bytes": VOICE_ENGINES[0]["asset_bytes"],
                "weights_bytes": VOICE_ENGINES[0]["weights_bytes"],
                "runtime_estimate_bytes":
                    VOICE_ENGINES[0]["runtime_estimate_bytes"],
                "estimate": True,
            },
            "state": state,
            "usable": state in ("usable", "ready"),
            "ready": state == "ready",
            "generated": self._generated_once,
            "primed": self._prime_done,
            "busy": self._slots > 0,
            "worker": {"alive": worker_alive,
                       "pid": self._worker.process.pid if self._worker
                       else None},
            "last_error": self._last_error,
            "voice_choice_error": voice_error,
            "qualification": self._qualification_status(assets, accel),
        }

    def _qualification_status(self, assets: dict, accel: dict) -> dict:
        try:
            receipt = json.loads(
                (self.assets_dir() / _QUALIFICATION_NAME).read_text())
        except (OSError, ValueError):
            receipt = None
        absent = {"qualified": False, "receipt": None,
                  "reason": "no qualification receipt (audible listener "
                            "check, zero CPU tensor dispatches, latency "
                            "measurement)"}
        if not isinstance(receipt, dict):
            return absent
        model_pin = _model_pin()
        backend = _backend_provenance()
        checks = {
            "pack_revision": receipt.get("pack_revision")
                             == VOICE_ENGINES[0]["revision"],
            "model_hash": receipt.get("model_sha256") == model_pin["sha256"],
            "mlx_backend": (backend["verified"] == "match"
                            and receipt.get("mlx_backend")
                            == backend["identity"]),
            "assets_verified": assets["verified"],
            "accelerator": accel["available"],
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            reason_detail = backend.get("detail")
            return {"qualified": False, "receipt": receipt,
                    "reason": "qualification receipt stale or mismatched ("
                              + ", ".join(failed) + ")"
                              + (f"; {reason_detail}" if reason_detail
                                 and "mlx_backend" in failed else "")}
        return {"qualified": True, "receipt": receipt, "reason": None}

    def record_qualification(self, receipt: dict) -> dict:
        """Write a durable hardware qualification receipt (parent-owned run).

        Required facts: listener_verified true, cpu_tensor_dispatches zero,
        a non-empty latency_receipt, and receipt_source naming the run log.
        The mlx binary hash and pinned model hash are computed here, so the
        receipt qualifies only the exact binary+model pair it measured.
        """
        missing = [key for key in ("listener_verified", "cpu_tensor_dispatches",
                                   "latency_receipt", "receipt_source")
                   if key not in receipt]
        if missing:
            raise ValueError("qualification receipt missing: "
                             + ", ".join(missing))
        if receipt["listener_verified"] is not True \
                or receipt["cpu_tensor_dispatches"] != 0:
            raise ValueError("qualification receipt not passed: listener "
                             "check must be true and CPU dispatches zero")
        if not receipt["latency_receipt"] or not receipt["receipt_source"]:
            raise ValueError("qualification receipt needs a latency figure "
                             "and a receipt source")
        backend = _backend_provenance()
        if backend["verified"] != "match":
            raise ValueError(
                "cannot tie qualification to the mlx backend: "
                + (backend.get("detail") or backend["verified"]))
        model_pin = _model_pin()
        record = {
            "pack_revision": VOICE_ENGINES[0]["revision"],
            "model_sha256": model_pin["sha256"],
            "mlx_backend": backend["identity"],
            "mlx_backend_paths": {
                "libmlx": backend["libmlx_loaded_path"],
                "mlx_core": next(
                    (f["path"] for f in backend["files"]
                     if f["label"] == "mlx.core extension"), None),
            },
            "mlx_version": backend["mx_version"],
            "listener_verified": True,
            "cpu_tensor_dispatches": receipt["cpu_tensor_dispatches"],
            "latency_receipt": receipt["latency_receipt"],
            "receipt_source": receipt["receipt_source"],
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        assets = self.assets_dir()
        assets.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=assets, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
        os.replace(tmp_name, assets / _QUALIFICATION_NAME)
        return {"recorded": True, "path": str(assets / _QUALIFICATION_NAME)}

    def prepare(self, approve_download: bool, *, fetch=None) -> dict:
        """Download and hash-verify what the voice choices need: the default
        engine always, plus the engine owning a persisted voice choice so an
        explicit pick keeps working. Engines nobody chose stay undownloaded.
        """
        default = VOICE_ENGINES[0]
        try:
            saved = _voice_pack(self.current_voice())
        except VoiceError:
            saved = None
        wanted = [default]
        if saved is not None and saved["id"] != default["id"]:
            wanted.append(saved)
        total_bytes = sum(p["asset_bytes"] for p in wanted)
        results = []
        for pack in VOICE_ENGINES:
            assets = self.assets_dir_for(pack["id"])
            if pack not in wanted:
                results.append({"engine": pack["id"], "downloaded": False,
                                "verified": False})
                continue
            if assets.is_dir() and self._asset_status_for(assets, pack)["verified"]:
                results.append({"engine": pack["id"], "downloaded": True,
                                "verified": True, "path": str(assets),
                                "bytes": pack["asset_bytes"]})
                continue
            if not approve_download:
                reason = ("voice pack download requires explicit approval "
                          f"({total_bytes} bytes total, "
                          f"{pack['license']} license, "
                          f"revision {pack['revision']})")
                return {"downloaded": False, "verified": False,
                        "path": str(self.assets_dir()), "reason": reason,
                        "engines": [{"engine": p["id"], "downloaded": False,
                                     "verified": False} for p in VOICE_ENGINES]}
            results.append(self._prepare_pack(pack, assets, fetch or _default_fetch))
        default_result = next(r for r in results
                              if r["engine"] == default["id"])
        return {**default_result, "engines": results}

    def _prepare_pack(self, pack: dict, assets: Path, fetch) -> dict:
        assets.mkdir(parents=True, exist_ok=True)
        try:
            for entry in sorted(pack["files"], key=lambda e: e["bytes"]):
                dest = assets / entry["name"]
                if dest.is_file() and dest.stat().st_size == entry["bytes"] \
                        and _sha256_file(dest) == entry["sha256"]:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".part")
                try:
                    fetch(_URL_TEMPLATE.format(repo=pack["repo"],
                                               revision=pack["revision"],
                                               name=entry["name"]), tmp)
                    actual = _sha256_file(tmp)
                    if actual != entry["sha256"]:
                        raise VoiceError(
                            f"voice pack file {entry['name']}: sha256 mismatch "
                            f"(expected {entry['sha256']}, got {actual}); "
                            "delete the pack and retry the download")
                    os.replace(tmp, dest)
                finally:
                    tmp.unlink(missing_ok=True)
            for entry in pack["files"]:
                dest = assets / entry["name"]
                if not dest.is_file() or dest.stat().st_size != entry["bytes"]:
                    raise VoiceError(
                        f"voice pack file {entry['name']} failed verification")
                actual = _sha256_file(dest)
                if actual != entry["sha256"]:
                    raise VoiceError(
                        f"voice pack file {entry['name']}: sha256 mismatch "
                        f"(expected {entry['sha256']}, got {actual})")
                with _stat_identity_lock:
                    _stat_identity_cache[(str(assets), entry["name"])] = \
                        _identity(dest.stat())
            self._write_receipt(assets, pack)
        except VoiceError as exc:
            self._last_error = str(exc)
            raise
        except Exception as exc:
            self._last_error = f"voice pack download failed: {exc}"
            raise VoiceError(self._last_error) from exc
        return {"engine": pack["id"], "downloaded": True, "verified": True,
                "path": str(assets), "bytes": pack["asset_bytes"]}

    def _write_receipt(self, assets: Path, pack: dict | None = None) -> None:
        pack = pack if pack is not None else VOICE_ENGINES[0]
        receipt = {
            "pack_id": pack["id"],
            "repo": pack["repo"],
            "revision": pack["revision"],
            "license": pack["license"],
            "files": {entry["name"]: {"bytes": entry["bytes"],
                                      "sha256": entry["sha256"]}
                      for entry in pack["files"]},
            "verified_by": ("sha256 at download time and once per process "
                            "startup; between hashes, file identity "
                            "(size, mtimes, ctime, inode) is compared "
                            "in-memory only"),
        }
        payload = json.dumps(receipt, indent=2, sort_keys=True)
        fd, tmp_name = tempfile.mkstemp(dir=assets, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
        os.replace(tmp_name, assets / _RECEIPT_NAME)

    def _guard(self) -> None:
        pack = self._current_engine()
        deps = probe_dependencies(pack)
        conflicts = _constraint_conflicts(deps)
        if deps["missing"] or conflicts:
            raise VoiceDependencyMissingError(
                "voice dependencies unsatisfied: "
                + "; ".join([", ".join(deps["missing"])] + conflicts)
                + "; install the runtime listed in status()")
        accel = probe_accelerator()
        if not accel["available"]:
            raise AcceleratorUnavailableError(
                f"voice synthesis requires the GPU accelerator "
                f"(default device {accel['device']!r}: {accel['detail']}); "
                "no CPU fallback is permitted")
        assets = self._asset_status_for(self.assets_dir_for(pack["id"]), pack)
        if not assets["verified"]:
            raise VoiceAssetsMissingError(
                f"voice pack missing or unverified at {assets['path']}; "
                "run setup with download approval")

    def _acquire_slot(self) -> None:
        with self._slots_cond:
            if self._slots >= MAX_PENDING_REQUESTS:
                raise VoiceBusyError(
                    f"voice queue is full ({MAX_PENDING_REQUESTS} pending); "
                    "retry when a sentence finishes")
            self._slots += 1

    def _release_slot(self) -> None:
        with self._slots_cond:
            self._slots = max(0, self._slots - 1)
            self._slots_cond.notify_all()

    def _ensure_worker(self, pack_id: str | None = None) -> _WorkerHandle:
        if pack_id is None:
            pack_id = VOICE_ENGINES[0]["id"]
        handle = self._workers.get(pack_id)
        if handle is None:
            handle = _WorkerHandle(str(self.assets_dir_for(pack_id)), pack_id)
            self._workers[pack_id] = handle
        self._worker = handle
        return handle

    def _reset_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            self._workers.pop(getattr(worker, "pack_id", None), None)
            worker.stop()

    def synthesize_chunks(self, text: str,
                          cancel: threading.Event) -> Iterator[PcmChunk]:
        """Stream one sentence as bounded PCM16 chunks (coordinator feed).

        Model work is serialized: a second caller queues (bounded by
        MAX_PENDING_REQUESTS), holding its slot until the worker frees.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("synthesis requires non-empty text")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text exceeds {MAX_TEXT_CHARS} characters")
        if cancel.is_set():
            raise SynthesisCancelled("cancelled before synthesis started")
        self._guard()
        self._acquire_slot()
        req_id = None
        locked = False
        try:
            queue_deadline = time.monotonic() + GENERATION_DEADLINE_SECONDS
            while not self._worker_lock.acquire(timeout=0.2):
                if cancel.is_set():
                    raise SynthesisCancelled("cancelled while queued")
                if time.monotonic() > queue_deadline:
                    raise VoiceBusyError(
                        "voice worker busy beyond the request deadline")
            locked = True
            if cancel.is_set():
                raise SynthesisCancelled("cancelled while queued")
            worker = self._ensure_worker(self._current_engine()["id"])
            self._req_counter += 1
            req_id = self._req_counter
            settled = False
            voice = self.current_voice()
            try:
                worker.conn.send({"type": "speak", "id": req_id,
                                  "text": text, "voice": voice})
            except Exception as exc:
                self._reset_worker()
                raise VoiceError(
                    f"voice worker pipe broke before request: {exc}") from exc
            timeout = FIRST_MESSAGE_TIMEOUT
            deadline = time.monotonic() + GENERATION_DEADLINE_SECONDS
            while True:
                if cancel.is_set():
                    self._cancel_request(worker, req_id)
                    raise SynthesisCancelled("cancelled during synthesis")
                msg = (None if not worker.conn.poll(timeout)
                       else self._recv_or_exit(worker))
                if msg is None:
                    self._reset_worker()
                    raise VoiceError(
                        "voice worker stopped responding; it was reset and "
                        "the next request will reload the model")
                timeout = MESSAGE_TIMEOUT
                if msg.get("id") not in (None, req_id):
                    continue
                kind = msg.get("type")
                if kind == "chunk":
                    yield PcmChunk(msg["sample_rate"], msg["data"])
                    if time.monotonic() > deadline:
                        self._reset_worker()
                        raise VoiceError(
                            "synthesis exceeded the "
                            f"{GENERATION_DEADLINE_SECONDS:.0f}s request "
                            "deadline; worker reset")
                    continue
                if kind == "loaded":
                    self._sample_rate = msg["sample_rate"]
                    continue
                if kind == "done":
                    self._generated_once = True
                    self._last_error = None
                    settled = True
                    return
                if kind == "cancelled":
                    settled = True
                    raise SynthesisCancelled("cancelled during synthesis")
                if kind == "worker_exit":
                    settled = True
                    self._reset_worker()
                    raise VoiceError(
                        "voice worker exited mid-request; it will reload on "
                        "the next request")
                settled = True
                exc_type = _EXPORTED_ERRORS.get(msg.get("error_type"),
                                                VoiceError)
                raise exc_type(msg.get("message", "synthesis failed"))
        finally:
            if locked:
                try:
                    if req_id is not None and not settled:
                        self._abandon_request(req_id)
                finally:
                    self._worker_lock.release()
            self._release_slot()

    @staticmethod
    def _recv_or_exit(worker: _WorkerHandle):
        try:
            return worker.conn.recv()
        except EOFError:
            return {"type": "worker_exit"}

    def _cancel_request(self, worker: _WorkerHandle, req_id: int) -> None:
        """Cooperative cancel; kill+reset if the worker misses the grace."""
        try:
            worker.conn.send({"type": "cancel", "id": req_id})
        except Exception:
            pass
        grace_end = time.monotonic() + CANCEL_GRACE
        while time.monotonic() < grace_end:
            if not worker.conn.poll(0.1):
                continue
            msg = self._recv_or_exit(worker)
            if msg.get("id") in (None, req_id) and msg.get("type") in (
                    "cancelled", "done", "error", "worker_exit"):
                if msg["type"] == "worker_exit":
                    self._reset_worker()
                return
        self._reset_worker()

    def _abandon_request(self, req_id: int) -> None:
        """Caller stopped consuming (close/GeneratorExit/cancel): settle it."""
        worker = self._worker
        if worker is None or not worker.process.is_alive():
            return
        self._cancel_request(worker, req_id)

    def synthesize(self, text: str, cancel: threading.Event) -> bytes:
        """Whole-sentence compatibility wrapper over synthesize_chunks."""
        rate = None
        samples = array.array("h")
        for chunk in self.synthesize_chunks(text, cancel):
            if rate is None:
                rate = chunk.sample_rate
            samples.frombytes(chunk.data)
        if rate is None or not samples:
            raise VoiceError("synthesis produced no audio")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(samples.tobytes())
        return buffer.getvalue()

    def prime(self, cancel: threading.Event) -> bool:
        """Pre-warm the default engine: start the resident worker and run
        one tiny synthesis so model load and first-infer warmup are paid
        before real speech is requested. Best effort: returns False on any
        refusal instead of raising; a real speak behaves exactly as before.
        """
        try:
            self._guard()
        except VoiceError:
            return False
        try:
            for _ in self.synthesize_chunks("Ready.", cancel):
                pass
        except (SynthesisCancelled, VoiceError):
            return False
        if self._generated_once:
            self._prime_done = True
        return self._prime_done

    def close(self) -> None:
        """Release worker residency; returns only once each child is dead."""
        with self._worker_lock:
            active, self._worker = self._worker, None
            residents = list(self._workers.values())
            self._workers.clear()
        for worker in dict.fromkeys(([active] if active else []) + residents):
            if not worker.stop():
                raise VoiceError("voice worker could not be confirmed dead")
        self._last_error = None
