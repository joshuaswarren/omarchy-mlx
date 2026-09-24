#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""mlx-omarchy-parakeet — installed Parakeet reference product.

`download` fetches the pinned public reference (lock:
``overlay/tools/coreml/parakeet-reference.lock``) into the shared cache
and verifies every file's SHA-256 before and after. Nothing is trusted
on presence or mtime; the receipt (``--json``) records the exact bytes
on disk.

`transcribe` runs the pinned reference end to end on the installed
runtime: FLAC decode, Vulkan mel frontend, ANE island encoder through
the shipped worker and strict libane, GPU-resident greedy TDT decode,
detokenization. The installed asset hashes and the frozen acceptance
pins live in ``share/mlx-omarchy/parakeet-1/parakeet-runtime-pin.json``;
any mismatch, missing capability, or divergent output is an explicit
refusal — there is no CPU or GPU-only encoder fallback.
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
    return code


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


def _verify_assets(pin: dict) -> None:
    share = _share_dir()
    for name, files in pin["assets"]["bundles"].items():
        bundle_dir = share / "bundles" / name
        if not bundle_dir.is_dir():
            raise TranscribeRefusal(f"installed bundle {name} is missing")
        for relative, expected in sorted(files.items()):
            path = bundle_dir / relative
            actual = _sha256_file(path) if path.is_file() else None
            if actual != expected:
                raise TranscribeRefusal(
                    f"installed bundle {name}/{relative} does not match the "
                    f"pin: expected {expected}, found {actual}; refusing to "
                    f"execute unverified ANE programs"
                )
    for relative, expected in sorted(pin["assets"]["libane"].items()):
        path = share / "libane" / relative
        actual = _sha256_file(path) if path.is_file() else None
        if actual != expected:
            raise TranscribeRefusal(
                f"installed {relative} does not match the pin: expected "
                f"{expected}, found {actual}; refusing to load unverified "
                f"ANE userspace"
            )


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


def _decode_flac(path: Path):
    """Return int16 PCM, sample rate, and the decoder that produced them."""
    try:
        import soundfile
    except ImportError:
        pass
    else:
        pcm, rate = soundfile.read(str(path), dtype="int16")
        import numpy as np

        return np.ascontiguousarray(pcm), int(rate), f"soundfile {soundfile.__version__}"

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    if int(stream["channels"]) != 1:
        raise TranscribeRefusal("the pinned fixture must be mono")
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-acodec",
         "pcm_s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    version = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    import numpy as np

    return np.frombuffer(raw, dtype="<i2"), int(stream["sample_rate"]), version


def _transcribe(args) -> int:
    import shutil
    import tempfile

    # The shipped default island path is resident-batch: one private
    # worker and one deadline-bounded batch per pass, the configuration
    # every green E2E battery ran. An exported value wins.
    os.environ.setdefault("ANE_ISLAND_MODE", "resident-batch")

    pin = _load_pin()
    _check_ane_capability()
    _verify_assets(pin)
    worker = _worker_path()
    share = _share_dir()
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
    if audio_sha != pin["e2e"]["audio_fixture_sha256"]:
        raise TranscribeRefusal(
            f"audio fixture sha256 {audio_sha} is not the pinned "
            f"{pin['e2e']['audio_fixture_sha256']}; the installed runtime "
            f"executes the pinned reference only"
        )

    out = Path(args.out) if args.out else (
        default_cache_root() / "transcriptions"
        / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    out.mkdir(parents=True, exist_ok=True)

    scratch_root = Path(tempfile.mkdtemp(prefix="parakeet-transcribe-",
                                         dir=default_cache_root()))
    passed = False
    try:
        passed = _run_pipeline(args, pin, lock, cache_dir, fixture, audio_sha,
                               worker, share, scratch_root, out)
    finally:
        if not args.keep_scratch:
            shutil.rmtree(scratch_root, ignore_errors=True)
    return 0 if passed else 1


def _run_pipeline(args, pin, lock, cache_dir, fixture, audio_sha, worker,
                  share, scratch_root, out) -> bool:
    """mel -> ANE islands -> TDT -> transcript, then the pin checks."""
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
        pcm, rate, decoder_name = _decode_flac(fixture)
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

    # --------------------------------------------------------------- 2. mel
    def stage_mel():
        result = extract_chunk_features(waveform)
        mx.eval(result.mel, result.mask, result.encoder_features,
                result.encoder_mask)
        return result

    mel_result = stage("mel_frontend", trace_snapshot, stage_mel)

    # ----------------------------------------------------------- 3. encoder
    deadline_ms = int(args.deadline_ms)
    island = encoder_module.AneIsland(
        worker, share / "libane" / "libane-strict.so",
        share / "bundles", scratch_root, deadline_ms,
    )
    source = _ensure_encoder_source(lock, pin, cache_dir)
    runner = encoder_module.EncoderRunner(
        source / "model.mil", source / "model-root", island,
    )

    def stage_encoder():
        try:
            return runner.run(
                inputs={
                    "input_features": mel_result.encoder_features,
                    "attention_mask": mel_result.encoder_mask,
                },
                wanted={"encoder_hidden", "encoder_mask"},
                stop_after="encoder_mask",
            )
        finally:
            island.close()

    encoded = stage("encoder_ane", trace_snapshot, stage_encoder)
    encoder_hidden = encoded["encoder_hidden"].astype(mx.float32)
    encoder_mask = encoded["encoder_mask"].astype(mx.int32)
    mx.eval(encoder_hidden, encoder_mask)

    # ------------------------------------------- 4. decoder, joint, control
    decoder = stage(
        "decoder_load", trace_snapshot,
        lambda: load_decoder(cache_dir / "decoder.mlpackage"),
    )
    fused_packed = pack_step_weights(decoder, cache_dir / "joint.mlpackage")
    counts = {"decoder_calls": 0, "joint_calls": 0}
    decoder_ns = 0
    joint_ns = 0
    frame_holder = [0]
    fused = {"frame": None, "state": None, "tok": None, "dur": None}

    def decoder_callback(token_id, current_hidden, current_cell):
        nonlocal decoder_ns
        counts["decoder_calls"] += 1
        started = time.monotonic_ns()
        state_out, tok, dur, _, _ = run_step(
            fused_packed, current_hidden, current_cell, token_id,
            encoder_hidden, frame_holder[0],
        )
        mx.eval(state_out)
        decoder_ns += time.monotonic_ns() - started
        dec_state = state_out[0:640].reshape(1, 640)
        fused["frame"] = frame_holder[0]
        fused["state"] = dec_state
        fused["tok"] = tok
        fused["dur"] = dur
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
        state_out, tok, dur, _, _ = run_step(
            fused_packed, None, None, 0, encoder_hidden, frame_index,
            skip_lstm=True, dec_in=decoder_state,
        )
        mx.eval(state_out)
        joint_ns += time.monotonic_ns() - started
        return JointDecision(tok, dur)

    def stage_tdt():
        with mx.stream(mx.gpu):
            hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
            cell = mx.zeros((2, 1, 640), dtype=mx.float32)
        mx.eval(hidden, cell)
        return tdt_decode(
            packed=fused_packed,
            encoder=encoder_hidden,
            valid_frames=int(encoder_hidden.shape[1]),
            config=lock.tdt,
            initial_hidden=hidden,
            initial_cell=cell,
            run_decoder=decoder_callback,
            run_joint=joint_callback,
        )

    tdt = stage("tdt_decode", trace_snapshot, stage_tdt)

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

    mel_host = np.asarray(mel_result.mel)
    hidden_host = np.asarray(encoder_hidden).astype(np.float32)
    actual_tokens = list(tdt.token_ids)
    native_tokens = list(expected["token_ids"])
    transcript_sha = hashlib.sha256(transcript.encode()).hexdigest()

    # The whole-program ANE pipeline (all 13701 ops in the ANE, fp16) and the
    # island hybrid (1230 ops on the GPU, fp32) differ in the f32 tensor's
    # low bits; transcript, tokens, durations and frame indices are
    # identical. The pin carries one hidden sha per encoder path.
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
    check("cpu_tensor_events",
          runner.cpu_tensor_events == expected["cpu_tensor_events"],
          {"expected": expected["cpu_tensor_events"],
           "actual": runner.cpu_tensor_events})
    check("decode_control",
          tdt.decode_path == expected["decode_control"]
          and tdt.fallback_reason is None,
          {"expected": expected["decode_control"],
           "actual": tdt.decode_path, "fallback_reason": tdt.fallback_reason})
    check("finite_hidden",
          int(np.isnan(hidden_host).sum()) == contract.nan_count_allowed
          and int(np.isinf(hidden_host).sum()) == contract.inf_count_allowed,
          {"nan": int(np.isnan(hidden_host).sum()),
           "inf": int(np.isinf(hidden_host).sum())})

    np.save(out / "waveform.npy", np.asarray(waveform))
    np.save(out / "mel.npy", mel_host)
    np.save(out / "encoder_hidden.npy", hidden_host)
    np.save(out / "encoder_mask.npy", np.asarray(encoder_mask).astype(np.int32))
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
            "valid_encoder_frames": int(encoder_hidden.shape[1]),
            "control": tdt.decode_path,
            "tdt_fallback_reason": tdt.fallback_reason,
        },
        "ane": {
            "submissions": island.submissions,
            "worker_starts": island.worker_starts,
            "timeouts": island.timeouts,
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
    return passed


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
        "verify", help="verify the cache against the lock and pin"
    )
    verify.set_defaults(handler=_verify)

    transcribe = sub.add_parser(
        "transcribe",
        help="run the pinned reference end to end (mel, ANE encoder, TDT)",
    )
    transcribe.add_argument(
        "audio", nargs="?", default=None,
        help="path to the pinned audio fixture (default: the cached copy)",
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
