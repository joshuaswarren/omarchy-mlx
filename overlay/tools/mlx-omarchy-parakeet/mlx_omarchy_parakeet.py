#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""mlx-omarchy-parakeet — installed Parakeet reference product.

`download` fetches the pinned public reference (lock:
``overlay/tools/coreml/parakeet-reference.lock``) into the shared cache
and verifies every file's SHA-256 before and after. Nothing is trusted
on presence or mtime; the receipt (``--json``) records the exact bytes
on disk.

`transcribe` runs the end-to-end pipeline on the installed runtime:
FLAC/audio decode (any mono-friendly input; 16 kHz mono is native,
anything else is downmixed/resampled through ffmpeg), Vulkan mel
frontend, ANE island encoder through the shipped worker and strict
libane, greedy TDT decode (host control loop by default;
``MLX_OMARCHY_TDT_HOST`` set to a false value opts into the one-dispatch
GPU loop), detokenization. The pinned reference fixture runs the golden
contract: every stage output is checked against the frozen acceptance
pins. Any other audio runs the general contract (on-device execution,
finite outputs, stream geometry, cross-repeat determinism); audio longer
than the 30 s model window is transcribed in 30 s chunks with carried
TDT recurrent state. Any mismatch, missing capability, or divergent
output is an explicit refusal — there is no CPU or GPU-only encoder
fallback.
"""

import argparse
import contextlib
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import time
from pathlib import Path

_TOOLS = str(Path(__file__).resolve().parents[1])
_COREML = str(Path(__file__).resolve().parents[1] / "coreml")
for _entry in (_TOOLS, _COREML):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from coreml import fetch_parakeet_reference as fetch  # noqa: E402
from coreml.reference import (  # noqa: E402
    ReferenceError,
    ReferenceLock,
    default_cache_root,
    model_cache_dir,
)

RECEIPT_SCHEMA = "mlx-omarchy.parakeet-download-receipt.v1"
REPORT_SCHEMA = "mlx-omarchy.parakeet-transcribe.v1"


class TranscribeRefusal(RuntimeError):
    """The installed runtime cannot or must not run; the reason is named."""


_TRANSCRIBE_DEPS = ("numpy", "google.protobuf")


def _check_runtime_deps() -> None:
    """Refuse with the exact install line when deps are missing.

    The wheel deliberately declares no hard dependencies (the upstream
    packaging contract), so the product names them instead of failing
    with an import traceback mid-run. soundfile is optional: FLAC
    decode falls back to ffmpeg when it is absent.
    """
    import importlib

    missing = []
    for module in _TRANSCRIBE_DEPS:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise TranscribeRefusal(
            "missing runtime dependencies for transcribe: "
            f"{', '.join(missing)}; install them with "
            "`pip install numpy protobuf` "
            "(ffmpeg handles FLAC decode when soundfile is absent)"
        )


def _bin_dir() -> Path:
    return Path(__file__).resolve().parent


def _share_dir() -> Path:
    return _bin_dir().parent / "share" / "mlx-omarchy" / "parakeet-1"


def _cache_dir(lock: ReferenceLock) -> Path:
    return model_cache_dir(
        default_cache_root(), lock.model_repo, lock.model_revision
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _receipt(lock: ReferenceLock, cache_dir: Path, elapsed_ms: int,
             fixture_path: Path | None) -> dict:
    return {
        "schema": RECEIPT_SCHEMA,
        "model_repo": lock.model_repo,
        "model_revision": lock.model_revision,
        "reference_commit": lock.reference_commit,
        "cache_dir": str(cache_dir),
        "files": [
            {
                "path": f.path,
                "size": f.size,
                "sha256": f.sha256,
                "verified": (cache_dir / f.path).is_file()
                and fetch.sha256_file(cache_dir / f.path) == f.sha256,
            }
            for f in lock.files
        ],
        "audio_fixture": {
            "url": lock.audio.url,
            "sha256": lock.audio.sha256,
            "size": lock.audio.size,
            "cached": fixture_path is not None
            and fixture_path.is_file()
            and _sha256_file(fixture_path) == lock.audio.sha256,
        },
        "elapsed_ms": elapsed_ms,
    }


def _fixture_path(cache_dir: Path) -> Path:
    return cache_dir / "audio" / "fixture.flac"


def _fetch_fixture(lock: ReferenceLock, cache_dir: Path) -> None:
    """Fetch the pinned audio fixture when the cache does not hold it."""
    dest = _fixture_path(cache_dir)
    if dest.is_file() and _sha256_file(dest) == lock.audio.sha256:
        return
    print(f"fetching audio fixture ({lock.audio.size} bytes)")
    fetch._download_to(lock.audio.url, dest)
    actual = _sha256_file(dest)
    if actual != lock.audio.sha256:
        dest.unlink(missing_ok=True)
        raise ReferenceError(
            f"fetched audio fixture does not match the pin: "
            f"expected {lock.audio.sha256}, got {actual} (refusing to use it)"
        )


def _download(args) -> int:
    lock = ReferenceLock.load()
    cache_dir = _cache_dir(lock)
    started = time.monotonic()
    # Human-readable progress goes to stderr so --json stdout stays
    # parseable.
    with contextlib.redirect_stdout(sys.stderr):
        code = fetch.cmd_download(lock, cache_dir, force=args.force)
        if code == 0:
            _fetch_fixture(lock, cache_dir)
    elapsed = int((time.monotonic() - started) * 1000)
    if args.json:
        print(json.dumps(
            _receipt(lock, cache_dir, elapsed, _fixture_path(cache_dir)),
            indent=2,
        ))
    return code


def _verify(args) -> int:
    lock = ReferenceLock.load()
    cache_dir = _cache_dir(lock)
    code = fetch.cmd_verify(lock, cache_dir)
    fixture = _fixture_path(cache_dir)
    if fixture.is_file():
        if _sha256_file(fixture) == lock.audio.sha256:
            print(f"OK: audio fixture verified in {fixture}")
        else:
            print(f"MISMATCH: audio fixture {fixture} does not match the pin",
                  file=sys.stderr)
            return 1
    else:
        print("audio fixture not cached yet; run `download`", file=sys.stderr)
        return 1
    if code != 0:
        return code

    # Golden e2e verify: run the pinned reference end to end with the full
    # per-item pin checks (mel/hidden sha, emissions, token ids, transcript).
    # Only possible where the ANE runtime can execute; byte verification
    # above stays meaningful on any host.
    try:
        _check_ane_capability()
    except TranscribeRefusal as error:
        print(f"NOTE: golden e2e verify skipped: {error}", file=sys.stderr)
        return 0
    e2e = argparse.Namespace(audio=None, out=None, deadline_ms=20000,
                             repeat=1, keep_scratch=False)
    print("running golden e2e verify (pinned fixture, full pin checks)")
    return 0 if _transcribe(e2e) == 0 else 1


def _load_pin() -> dict:
    pin_path = _share_dir() / "parakeet-runtime-pin.json"
    if not pin_path.is_file():
        raise TranscribeRefusal(
            f"the Parakeet runtime assets are not installed "
            f"(expected {pin_path}); this wheel does not ship the ANE "
            f"runtime — the pinned bundles and libane install on aarch64 "
            f"hosts only"
        )
    return json.loads(pin_path.read_text())


def _check_ane_capability() -> None:
    """Refuse unless the host can run the ANE island encoder."""
    value = platform.system().lower()
    if value != "linux":
        raise TranscribeRefusal(
            f"the ANE runtime requires Linux (Asahi); found {platform.system()}"
        )
    machine = platform.machine().lower()
    if machine not in ("aarch64", "arm64"):
        raise TranscribeRefusal(
            f"the Apple Neural Engine exists only on aarch64 Apple "
            f"silicon hosts; found {machine}"
        )
    kill = _env_off("MLX_OMARCHY_ANE_DEVICE")
    if kill:
        raise TranscribeRefusal(
            "ANE disabled by MLX_OMARCHY_ANE_DEVICE=off; the installed "
            "product has no non-ANE encoder path, so transcribe refuses "
            "instead of falling back"
        )
    accel = Path(os.environ.get("MLX_OMARCHY_ACCEL_DEV", "/dev/accel/accel0"))
    if not accel.exists():
        raise TranscribeRefusal(
            f"no Apple ANE device node at {accel}; load the asahiANE/ane "
            f"driver or point MLX_OMARCHY_ACCEL_DEV at the accel device"
        )
    if not stat.S_ISCHR(os.stat(accel).st_mode):
        raise TranscribeRefusal(
            f"{accel} is not a character device "
            f"(mode {oct(os.stat(accel).st_mode)}); the ANE driver is "
            f"not bound"
        )
    if not Path("/sys/module/ane").is_dir():
        raise TranscribeRefusal(
            "the ane kernel module is not loaded (/sys/module/ane missing)"
        )


def _env_off(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("off", "0", "false", "no")


# Installed-asset verification lives at the worker's load boundary: the
# resident session seals every consumed file, hashes the sealed bytes,
# and refuses any mismatch with pin["assets"] before device load.


def _worker_path() -> Path:
    worker = _bin_dir() / "mlx-omarchy-ane-worker"
    if not worker.is_file():
        raise TranscribeRefusal(
            f"the ANE worker is not installed (expected {worker}); this "
            f"wheel does not ship the ANE runtime — aarch64 wheels build "
            f"it with MLX_OMARCHY_ANE_DEVICE"
        )
    return worker


def _ensure_encoder_source(lock: ReferenceLock, pin: dict, cache_dir: Path) -> Path:
    """The depalettized textual-MIL encoder source, emitted once, verified."""
    source = cache_dir / "encoder-source" / lock.model_revision
    mil = source / "model.mil"
    expected = pin["encoder_source"]["mil_sha256"]
    if mil.is_file():
        actual = _sha256_file(mil)
        if actual != expected:
            raise TranscribeRefusal(
                f"cached encoder source {mil} does not match the pin: "
                f"expected {expected}, found {actual}; remove {source} and "
                f"re-run to re-emit, or investigate the drift"
            )
        return source
    from coreml.mil_adapter import AdapterError, emit_mlpackage

    source.parent.mkdir(parents=True, exist_ok=True)
    print(f"emitting encoder source into {source} (one-time, CPU-bound)")
    try:
        emit_mlpackage(
            cache_dir / "encoder.mlpackage", source,
            Path(__file__).resolve().parents[1] / "coreml"
            / "parakeet-reference.lock",
        )
    except AdapterError as error:
        raise TranscribeRefusal(f"encoder source emission failed: {error}") from error
    actual = _sha256_file(mil)
    if actual != expected:
        raise TranscribeRefusal(
            f"emitted encoder source does not match the pin: expected "
            f"{expected}, found {actual}; refusing to run an unpinned graph"
        )
    return source


def _decode_audio(path: Path):
    """Return int16 mono 16 kHz PCM, sample rate, and the decoder used.

    The fast path is libsndfile (soundfile) when the file already is mono
    16 kHz. Anything else (stereo, other rates, other containers) goes
    through ffmpeg's deterministic `-ar 16000 -ac 1` downmix/resample so
    general audio is usable; a native 16 kHz mono file keeps the exact
    legacy decode (no resampler is engaged for it).
    """
    try:
        import soundfile
    except ImportError:
        pass
    else:
        pcm, rate = soundfile.read(str(path), dtype="int16")
        import numpy as np

        if int(rate) == 16000 and getattr(pcm, "ndim", 1) == 1:
            return np.ascontiguousarray(pcm), int(rate), f"soundfile {soundfile.__version__}"

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    rate, channels = int(stream["sample_rate"]), int(stream["channels"])
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le",
           "-acodec", "pcm_s16le"]
    if rate != 16000 or channels != 1:
        cmd += ["-ar", "16000", "-ac", "1"]
    raw = subprocess.run(cmd + ["-"], capture_output=True, check=True).stdout
    version = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    import numpy as np

    return np.frombuffer(raw, dtype="<i2"), 16000, version


def _transcribe(args) -> int:
    import shutil
    import tempfile

    # The shipped default island path is resident-batch: one private
    # worker and one deadline-bounded batch per pass, the configuration
    # every green E2E battery ran. An exported value wins.
    os.environ.setdefault("ANE_ISLAND_MODE", "resident-batch")

    pin = _load_pin()
    _check_ane_capability()
    worker = _worker_path()
    share = _share_dir()
    # Installed prefix layouts resolve the whole-encoder bundle from this
    # CLI's own share (the discovery fallback cannot see a prefix's share
    # through a symlinked coreml/ and would silently take the split-island
    # path). An explicit MLX_OMARCHY_WHOLE_ENCODER_BUNDLE still wins.
    _installed_whole = share / "bundles" / "parakeet-encoder-whole"
    if (_installed_whole / "manifest.json").is_file():
        os.environ.setdefault(
            "MLX_OMARCHY_WHOLE_ENCODER_BUNDLE", str(_installed_whole)
        )
    _check_runtime_deps()

    lock = ReferenceLock.load()
    cache_dir = _cache_dir(lock)
    ok, mismatches = fetch.verify_cache(cache_dir, lock)
    if not ok:
        raise TranscribeRefusal(
            "the reference cache does not verify against the lock; run "
            "`mlx-omarchy-parakeet download` first "
            f"({len(mismatches)} mismatched/missing files in {cache_dir})"
        )
    fixture = Path(args.audio) if args.audio else _fixture_path(cache_dir)
    if not fixture.is_file():
        raise TranscribeRefusal(
            f"audio fixture {fixture} is not cached; run "
            f"`mlx-omarchy-parakeet download` first"
        )
    audio_sha = _sha256_file(fixture)

    repeat = max(1, int(getattr(args, "repeat", 1) or 1))
    pinned = audio_sha == pin["e2e"]["audio_fixture_sha256"]
    passed = True
    prior: tuple[list[int], str] | None = None
    for index in range(repeat):
        out = Path(args.out) if args.out else (
            default_cache_root() / "transcriptions"
            / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        )
        if repeat > 1:
            out = out.with_name(f"{out.name}-r{index + 1}")
        out.mkdir(parents=True, exist_ok=True)

        scratch_root = Path(tempfile.mkdtemp(prefix="parakeet-transcribe-",
                                             dir=default_cache_root()))
        try:
            ok, tokens, transcript = _run_pipeline(
                args, pin, lock, cache_dir, fixture, audio_sha, worker,
                share, scratch_root, out, pinned, prior)
            if not ok:
                passed = False
            prior = (tokens, transcript)
        finally:
            if not args.keep_scratch:
                shutil.rmtree(scratch_root, ignore_errors=True)
    return 0 if passed else 1


def _run_pipeline(args, pin, lock, cache_dir, fixture, audio_sha, worker,
                  share, scratch_root, out, pinned, prior) -> tuple:
    """mel -> ANE islands -> TDT -> transcript, then the checks.

    `pinned` selects the golden contract (per-item sha equality against
    pin["e2e"]); any other audio takes the general contract (finite,
    on-device, mask-consistent, deterministic across repeats). Returns
    (passed, token_ids, transcript).
    """
    import numpy as np

    import mlx.core as mx
    from importlib.metadata import distribution

    from coreml.parakeet_tdt import DecoderStep, JointDecision, tdt_decode
    from coreml.tokenizer import ParakeetTokenizer
    from coreml.vulkan_decoder import load_decoder
    from coreml.vulkan_decoder_step import pack_step_weights, run_step
    from coreml.vulkan_mel import extract_chunk_features, trace_snapshot
    from coreml import vulkan_encoder as encoder_module

    mx.set_default_device(mx.gpu)

    contract = lock.numerical_contract
    if contract is None:
        raise TranscribeRefusal("the reference lock has no frozen numerical contract")
    tokenizer_sha = next(
        item.sha256 for item in lock.files if item.path == "tokenizer.json"
    )
    expected = pin["e2e"]

    stages_records: list[dict] = []

    def stage(name: str, snapshot, work):
        before = snapshot()
        started = time.monotonic_ns()
        result = work()
        elapsed = time.monotonic_ns() - started
        after = snapshot()
        stages_records.append({
            "stage": name,
            "wall_ns": elapsed,
            "wall_ms": round(elapsed / 1e6, 3),
            "gpu_counter_delta": {k: after[k] - before[k] for k in after},
        })
        return result

    # ------------------------------------------------------------- 1. audio
    def stage_audio():
        pcm, rate, decoder_name = _decode_audio(fixture)
        if rate != lock.audio.sample_rate:
            raise TranscribeRefusal(
                f"fixture sample rate {rate} != {lock.audio.sample_rate}"
            )
        with mx.stream(mx.gpu):
            waveform = mx.array(pcm).astype(mx.float32) / 32768.0
        mx.eval(waveform)
        return waveform, pcm.size, decoder_name

    waveform, sample_count, decoder_name = stage("audio_load", trace_snapshot,
                                                 stage_audio)

    # ------------------------------------------- 2. chunks (30 s windows)
    from coreml.vulkan_mel import CHUNK_SAMPLES

    if sample_count <= 0:
        raise TranscribeRefusal("audio decodes to zero samples")
    chunk_starts = list(range(0, int(sample_count), CHUNK_SAMPLES))
    single = len(chunk_starts) == 1

    # ----------------------------------------------------------- 3. encoder
    deadline_ms = int(args.deadline_ms)
    island = encoder_module.AneIsland(
        worker, share / "libane" / "libane-strict.so",
        share / "bundles", scratch_root, deadline_ms,
        # Every consumed byte is sealed at session open and bound to
        # pin["assets"] before device load; restarts re-authenticate.
        seal_assets=pin["assets"],
    )
    source = _ensure_encoder_source(lock, pin, cache_dir)
    runner = encoder_module.EncoderRunner(
        source / "model.mil", source / "model-root", island,
    )

    # ------------------------------------------- 4. decoder, joint, control
    decoder = stage(
        "decoder_load", trace_snapshot,
        lambda: load_decoder(cache_dir / "decoder.mlpackage"),
    )
    fused_packed = pack_step_weights(decoder, cache_dir / "joint.mlpackage")
    import numpy as np

    from coreml.vulkan_decoder_step import _JOINT_OUT, _VOCAB, _WINDOW_SLOTS

    counts = {"decoder_calls": 0, "joint_calls": 0}
    decoder_ns = 0
    joint_ns = 0
    frame_holder = [0]
    enc_holder = [None]
    valid_frames = 0
    fused = {"frame": None, "state": None, "tok": None, "dur": None}
    # Speculative joint window: the decoder step's submit also evaluates the
    # joint head for the next `spec_frames` frames against the new state,
    # so frame-entry joints (blank runs, max-symbol roll-over) resolve from
    # already-synced logits without another submit.
    spec = {"base": None, "state": None, "host": None}
    spec_frames = _WINDOW_SLOTS

    def _begin_chunk(encoder_hidden):
        enc_holder[0] = encoder_hidden
        frame_holder[0] = 0
        fused.update(frame=None, state=None, tok=None, dur=None)
        spec.update(base=None, state=None, host=None)
        return int(encoder_hidden.shape[1])

    def _spec_decision(frame_index, decoder_state):
        host = spec["host"]
        if host is None or spec["state"] is not decoder_state:
            return None
        offset = frame_index - spec["base"]
        if not 0 <= offset < host.shape[0]:
            return None
        row = host[offset]
        return JointDecision(
            int(np.argmax(row[:_VOCAB])),
            int(np.argmax(row[_VOCAB:_JOINT_OUT])),
        )

    def decoder_callback(token_id, current_hidden, current_cell):
        nonlocal decoder_ns
        counts["decoder_calls"] += 1
        started = time.monotonic_ns()
        state_out, tok, dur, _, _, window = run_step(
            fused_packed, current_hidden, current_cell, token_id,
            enc_holder[0], frame_holder[0],
            spec_frames=spec_frames, spec_valid=valid_frames,
        )
        mx.eval(state_out)
        decoder_ns += time.monotonic_ns() - started
        dec_state = state_out[0:640].reshape(1, 640)
        fused["frame"] = frame_holder[0]
        fused["state"] = dec_state
        fused["tok"] = tok
        fused["dur"] = dur
        spec["base"] = frame_holder[0] + 1
        spec["state"] = dec_state
        # Materialize the window on the host now: the per-row slice in
        # _spec_decision would otherwise be a lazy GPU copy paying its own
        # dispatch + submit per frame-entry joint.
        spec["host"] = np.asarray(window) if window is not None else None
        return DecoderStep(
            dec_state,
            state_out[640:1920].reshape(2, 1, 640),
            state_out[1920:3200].reshape(2, 1, 640),
        )

    def joint_callback(frame_index, decoder_state):
        nonlocal joint_ns
        counts["joint_calls"] += 1
        frame_holder[0] = frame_index
        started = time.monotonic_ns()
        if fused["frame"] == frame_index and fused["state"] is decoder_state:
            joint_ns += time.monotonic_ns() - started
            return JointDecision(fused["tok"], fused["dur"])
        decision = _spec_decision(frame_index, decoder_state)
        if decision is not None:
            joint_ns += time.monotonic_ns() - started
            return decision
        state_out, tok, dur, _, _, window = run_step(
            fused_packed, None, None, 0, enc_holder[0], frame_index,
            skip_lstm=True, dec_in=decoder_state,
            spec_frames=spec_frames, spec_valid=valid_frames,
        )
        mx.eval(state_out)
        joint_ns += time.monotonic_ns() - started
        spec["base"] = frame_index + 1
        spec["state"] = decoder_state
        spec["host"] = np.asarray(window) if window is not None else None
        return JointDecision(tok, dur)

    def stage_mel(chunk_wave):
        def work():
            result = extract_chunk_features(chunk_wave)
            mx.eval(result.mel, result.mask, result.encoder_features,
                    result.encoder_mask)
            return result
        return work

    def stage_encoder(mel_result):
        return lambda: runner.run(
            inputs={
                "input_features": mel_result.encoder_features,
                "attention_mask": mel_result.encoder_mask,
            },
            wanted={"encoder_hidden", "encoder_mask"},
            stop_after="encoder_mask",
        )

    def stage_tdt(hidden, cell, chunk_hidden, chunk_valid_frames):
        return lambda: tdt_decode(
            packed=fused_packed,
            encoder=chunk_hidden,
            valid_frames=chunk_valid_frames,
            config=lock.tdt,
            initial_hidden=hidden,
            initial_cell=cell,
            run_decoder=decoder_callback,
            run_joint=joint_callback,
        )

    mel_parts: list = []
    hidden_parts: list = []
    mask_parts: list = []
    token_ids: list[int] = []
    frame_indices: list[int] = []
    all_durations: list[int] = []
    prev_state: tuple = (None, None)
    decode_path = "host"
    fallback_reason = "no chunk ran"
    chain_final: int | None = None
    chain_slots: int | None = None
    frame_base = 0
    last_mask_valid = 0
    last_window = 0
    try:
        for ci, start in enumerate(chunk_starts):
            suffix = "" if single else f"#{ci}"
            chunk_wave = (waveform if single
                          else waveform[start:start + CHUNK_SAMPLES])
            mel_result = stage(f"mel_frontend{suffix}", trace_snapshot,
                               stage_mel(chunk_wave))
            encoded = stage(f"encoder_ane{suffix}", trace_snapshot,
                            stage_encoder(mel_result))
            encoder_hidden = encoded["encoder_hidden"].astype(mx.float32)
            encoder_mask = encoded["encoder_mask"].astype(mx.int32)
            mx.eval(encoder_hidden, encoder_mask)
            window = int(encoder_hidden.shape[1])
            mask_host_i = np.asarray(encoder_mask)
            mask_valid = int(mask_host_i.sum())
            # Decode-by-slicing assumes validity is a prefix (padding tail).
            # A mask with holes is a contract break, not a decode input.
            flat = mask_host_i.reshape(-1)
            if not (flat[:mask_valid].all()
                    and not flat[mask_valid:].any()):
                raise TranscribeRefusal(
                    "encoder mask validity is not a prefix; refusing to "
                    "slice-decode a mask with holes")
            # The pinned golden contract decodes the legacy full padded
            # window (its reference transcript includes the padded-tail
            # artifact). General audio decodes only the encoder frames the
            # mask marks valid — real signal, no padded-tail tokens.
            decode_frames = window if pinned else min(mask_valid, window)
            if decode_frames <= 0:
                raise TranscribeRefusal(
                    f"encoder mask marks zero valid frames ({mask_valid})")
            last_mask_valid = mask_valid
            last_window = window
            decode_hidden = (encoder_hidden if decode_frames == window
                             else encoder_hidden[:, :decode_frames, :])
            valid_frames = _begin_chunk(decode_hidden)

            # The CoreML reference (GreedyTDTDecoder.decode, called once per
            # chunk by Pipeline.swift) zeroes hidden/cell at the start of
            # every 30 s window and decodes while t < sum(encoderMask) —
            # chunks are independent decodes with concatenated streams.
            # Match that contract exactly.
            with mx.stream(mx.gpu):
                hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
                cell = mx.zeros((2, 1, 640), dtype=mx.float32)
            mx.eval(hidden, cell)
            chunk_tdt = stage(f"tdt_decode{suffix}", trace_snapshot,
                              stage_tdt(hidden, cell, decode_hidden,
                                        decode_frames))

            offset = frame_base
            token_ids += list(chunk_tdt.token_ids)
            frame_indices += [int(f) + offset for f in chunk_tdt.frame_indices]
            all_durations += list(chunk_tdt.durations)
            prev_state = (chunk_tdt.hidden, chunk_tdt.cell)
            decode_path = chunk_tdt.decode_path
            fallback_reason = chunk_tdt.fallback_reason
            chain_final = chunk_tdt.final_frame
            chain_slots = chunk_tdt.slots_used
            frame_base += decode_frames
            if single:
                mel_parts = [mel_result.mel]
                hidden_parts = [encoder_hidden]
                mask_parts = [encoder_mask]
            else:
                mel_parts.append(mel_result.mel)
                hidden_parts.append(encoder_hidden)
                mask_parts.append(encoder_mask)
    finally:
        island.close()

    from coreml.parakeet_tdt import TdtOutput

    tdt = TdtOutput(
        token_ids=token_ids,
        frame_indices=frame_indices,
        durations=all_durations,
        hidden=prev_state[0],
        cell=prev_state[1],
        decode_path=decode_path,
        fallback_reason=fallback_reason,
        final_frame=chain_final,
        slots_used=chain_slots,
    )
    mel_host = (np.asarray(mel_parts[0]) if single else
                np.concatenate([np.asarray(p) for p in mel_parts], axis=0))
    hidden_host = (np.asarray(hidden_parts[0]).astype(np.float32) if single else
                   np.concatenate([np.asarray(p) for p in hidden_parts],
                                  axis=1).astype(np.float32))
    mask_host = (np.asarray(mask_parts[0]).astype(np.int32) if single else
                 np.concatenate([np.asarray(p) for p in mask_parts],
                                axis=1).astype(np.int32))

    # ----------------------------------------------------------- 5. tokenizer
    def stage_tokenizer():
        tokenizer = ParakeetTokenizer.load(
            cache_dir / "tokenizer.json", expected_sha256=tokenizer_sha
        )
        return tokenizer, tokenizer.decode(tdt.token_ids)

    tokenizer, transcript = stage("detokenize", trace_snapshot, stage_tokenizer)

    # ------------------------------------------------------- 6. pin checks
    checks = []

    def check(name: str, passed: bool, detail) -> None:
        checks.append({"check": name, "pass": bool(passed), "detail": detail})

    actual_tokens = list(tdt.token_ids)
    transcript_sha = hashlib.sha256(transcript.encode()).hexdigest()

    if pinned:
        native_tokens = list(expected["token_ids"])

        # The whole-program ANE pipeline (all 13701 ops in the ANE, fp16) and
        # the island hybrid (1230 ops on the GPU, fp32) differ in the f32
        # tensor's low bits; transcript, tokens, durations and frame indices
        # are identical. The pin carries one hidden sha per encoder path.
        whole_path = island.whole_bundle is not None
        whole_sha = expected.get("encoder_hidden_sha256_whole")
        expected_hidden_sha = (whole_sha if (whole_path and whole_sha)
                               else expected["encoder_hidden_sha256"])

        check("mel_sha256", _npy_sha(mel_host) == expected["mel_sha256"],
              {"expected": expected["mel_sha256"], "actual": _npy_sha(mel_host)})
        check("encoder_hidden_sha256",
              _npy_sha(hidden_host) == expected_hidden_sha,
              {"expected": expected_hidden_sha,
               "actual": _npy_sha(hidden_host),
               "encoder_path": "whole-encoder" if whole_path else "islands"})
        check("emissions", len(actual_tokens) == expected["emissions"],
              {"expected": expected["emissions"], "actual": len(actual_tokens)})
        check("token_ids", actual_tokens == native_tokens,
              {"matching_prefix_length": _prefix_len(actual_tokens, native_tokens)})
        check("frame_indices",
              list(tdt.frame_indices) == list(expected["frame_indices"]), {})
        check("durations", list(tdt.durations) == list(expected["durations"]), {})
        check("transcript",
              transcript == expected["transcript"]
              and transcript_sha == expected["transcript_sha256"],
              {"expected_sha256": expected["transcript_sha256"],
               "actual_sha256": transcript_sha})
    else:
        # General-audio contract: on-device execution, finite outputs, a
        # token stream consistent with the window geometry, and — when the
        # caller runs repeats — determinism against the first pass. Golden
        # byte equality is the pinned fixture's contract (`verify`), not a
        # general-audio one.
        check("cpu_tensor_events", runner.cpu_tensor_events == 0,
              {"expected": 0, "actual": runner.cpu_tensor_events})
        check("decode_control",
              tdt.decode_path == expected["decode_control"]
              and tdt.fallback_reason is None,
              {"expected": expected["decode_control"],
               "actual": tdt.decode_path,
               "fallback_reason": tdt.fallback_reason})
        check("finite_hidden",
              int(np.isnan(hidden_host).sum()) == contract.nan_count_allowed
              and int(np.isinf(hidden_host).sum()) == contract.inf_count_allowed,
              {"nan": int(np.isnan(hidden_host).sum()),
               "inf": int(np.isinf(hidden_host).sum())})
        monotone = all(a <= b for a, b in
                       zip(tdt.frame_indices, tdt.frame_indices[1:]))
        in_window = all(0 <= f < frame_base for f in tdt.frame_indices)
        check("frame_stream",
              len(actual_tokens) == len(tdt.frame_indices)
              == len(tdt.durations) and monotone and in_window,
              {"tokens": len(actual_tokens),
               "frames": len(tdt.frame_indices),
               "durations": len(tdt.durations),
               "monotone": monotone, "in_window": in_window,
               "decoded_frames": frame_base, "chunks": len(chunk_starts)})
        all_valid = last_mask_valid == last_window * len(chunk_starts)
        check("decode_geometry",
              0 < frame_base <= last_window * len(chunk_starts)
              and frame_base == sum(
                  min(int(np.asarray(m).sum()), int(m.shape[-1]))
                  for m in mask_parts),
              {"decoded_frames": frame_base,
               "mask_valid": last_mask_valid,
               "window": last_window,
               # Both implementations (Linux mel frontend and the macOS
               # CoreML reference) emit an all-valid mask by design, so the
               # decoded frame count equals the window count; the mask is
               # not duration-aware anywhere in the reference lineage.
               "mask_all_valid": bool(all_valid)})
        check("durations_domain",
              all(d in lock.tdt.durations for d in tdt.durations),
              {"domain": list(lock.tdt.durations),
               "observed": sorted(set(tdt.durations))})
        if prior is not None:
            check("repeat_determinism",
                  actual_tokens == prior[0] and transcript == prior[1],
                  {"tokens_equal": actual_tokens == prior[0],
                   "transcript_equal": transcript == prior[1]})

    np.save(out / "waveform.npy", np.asarray(waveform))
    np.save(out / "mel.npy", mel_host)
    np.save(out / "encoder_hidden.npy", hidden_host)
    np.save(out / "encoder_mask.npy", mask_host)
    (out / "transcript.txt").write_text(transcript)
    (out / "token_ids.json").write_text(json.dumps({
        "token_ids": actual_tokens,
        "frame_indices": list(tdt.frame_indices),
        "durations": list(tdt.durations),
    }, indent=2) + "\n")

    libmlx = Path(distribution("mlx-omarchy").locate_file("mlx/lib/libmlx.so"))
    passed = all(item["pass"] for item in checks)
    report = {
        "schema": REPORT_SCHEMA,
        "mode": "golden" if pinned else "general",
        "status": "match" if passed else "diverged",
        "host": {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "mlx": {
            "version": mx.__version__,
            "device": str(mx.device_info().get("device_name", "")),
            "libmlx_path": str(libmlx),
            "libmlx_sha256": _sha256_file(libmlx),
            "core_path": mx.__file__,
            "core_sha256": _sha256_file(Path(mx.__file__)),
        },
        "inputs": {
            "audio": {
                "path": str(fixture),
                "sha256": audio_sha,
                "samples": int(sample_count),
                "sample_rate": lock.audio.sample_rate,
                "decoder": decoder_name,
                "chunks": len(chunk_starts),
                "chunk_samples": CHUNK_SAMPLES,
                "pinned_fixture": pinned,
            },
            "model": {
                "repo": lock.model_repo,
                "revision": lock.model_revision,
                "tokenizer_sha256": tokenizer.sha256,
                "tokenizer_vocab_size": tokenizer.vocab_size,
            },
            "encoder_source": {
                "mil_sha256": _sha256_file(source / "model.mil"),
                "root": str(source),
            },
        },
        "stages": stages_records,
        "timing": {
            "total_pipeline_ms": round(
                sum(r["wall_ns"] for r in stages_records) / 1e6, 3),
            "decoder_total_ms": round(decoder_ns / 1e6, 3),
            "joint_total_ms": round(joint_ns / 1e6, 3),
        },
        "execution": {
            "ane_mode": True,
            "encoder_ops_executed": runner.executed,
            "encoder_gpu_ops": runner.gpu_ops,
            "encoder_ane_ops": runner.ane_ops,
            "encoder_layers": runner.layers,
            "cpu_tensor_events": runner.cpu_tensor_events,
            "decoder_calls": counts["decoder_calls"],
            "joint_calls": counts["joint_calls"],
            # Scope honesty: these two counters increment only in the host
            # control-loop callbacks. On the default gpu-chain decode the
            # whole loop runs device-side and the honest work counters are
            # final_frame/slots_used below — zeros here mean "host loop not
            # taken", never "compute skipped" (the transcript stream proves
            # execution).
            "decode_counter_scope": ("host-callbacks" if tdt.decode_path
                                     == "host" else "device-chain"),
            "chain_final_frame": tdt.final_frame,
            "chain_slots_used": tdt.slots_used,
            "valid_encoder_frames": frame_base,
            "encoder_mask_valid_frames": last_mask_valid,
            "encoder_window_frames": last_window,
            "control": tdt.decode_path,
            "tdt_fallback_reason": tdt.fallback_reason,
        },
        "ane": {
            "submissions": island.submissions,
            "worker_starts": island.worker_starts,
            "timeouts": island.timeouts,
            # Session lifecycle, reported separately from per-call submit
            # latency: open covers worker spawn + bundle register + device
            # program load (0 when this pass reused the process-level
            # session), batch_open is this pass's ensure cost, close the
            # batch-scope release. Submit latency stays in exec_ms/log.
            "session": {
                "shared": island.share_session,
                "reused": island.session_reused,
                "open_ms": round(island.session_open_ns / 1e6, 3),
                "batch_open_ms": round(island.batch_open_ns / 1e6, 3),
                "close_ms": round(island.session_close_ns / 1e6, 3),
            },
            "input_bytes": island.input_bytes,
            "output_bytes": island.output_bytes,
            "exec_ns": island.exec_ns,
            "exec_ms": round(island.exec_ns / 1e6, 3),
            "worker": str(worker),
            "worker_sha256": _sha256_file(worker),
            "libane": str(share / "libane" / "libane-strict.so"),
            "libane_sha256": _sha256_file(share / "libane" / "libane-strict.so"),
            "bundles": sorted({record["bundle"] for record in island.log}),
            "log": island.log,
        },
        "verification": {"pin_schema": pin["schema"], "checks": checks},
        "transcript": transcript,
    }
    (out / "transcribe-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "checks_failed": [c["check"] for c in checks if not c["pass"]],
        "emissions": len(actual_tokens),
        "total_pipeline_ms": report["timing"]["total_pipeline_ms"],
        "out": str(out),
        "transcript": transcript,
    }, indent=2, ensure_ascii=False))
    return passed, actual_tokens, transcript


def _npy_sha(array) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _prefix_len(actual: list, native: list) -> int:
    length = 0
    for mine, theirs in zip(actual, native):
        if mine != theirs:
            break
        length += 1
    return length


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mlx-omarchy-parakeet")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser(
        "download", help="fetch and verify the pinned reference"
    )
    download.add_argument(
        "--force", action="store_true", help="re-fetch mismatched files"
    )
    download.add_argument(
        "--json", action="store_true", help="emit a machine-readable receipt"
    )
    download.set_defaults(handler=_download)

    verify = sub.add_parser(
        "verify",
        help="verify the cache against the lock and pin, then run the "
             "golden e2e (pinned fixture) where the ANE runtime exists",
    )
    verify.set_defaults(handler=_verify)

    transcribe = sub.add_parser(
        "transcribe",
        help="mel, ANE encoder and TDT over 16 kHz mono audio; the pinned "
             "fixture additionally runs the golden pin checks",
    )
    transcribe.add_argument(
        "audio", nargs="?", default=None,
        help="audio file to transcribe (default: the pinned fixture; any "
             "other audio takes the general contract, >30 s is chunked)",
    )
    transcribe.add_argument(
        "-o", "--out", default=None,
        help="output directory (default: a timestamped dir in the cache)",
    )
    transcribe.add_argument(
        "--deadline-ms", type=int, default=20000,
        help="per-submit ANE worker deadline in ms (default: 20000)",
    )
    transcribe.add_argument(
        "--repeat", type=int, default=1,
        help="transcriptions in this process; run 2+ reuses the resident "
             "session and reports startup separately (default: 1)",
    )
    transcribe.add_argument(
        "--keep-scratch", action="store_true",
        help="keep the ANE worker scratch directory after the run",
    )
    transcribe.set_defaults(handler=_transcribe)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReferenceError, TranscribeRefusal) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
