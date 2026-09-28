# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Arbitrary-audio dictation front end for the pinned Parakeet runtime.

The installed product's ``transcribe`` CLI executes the pinned reference
fixture only; its hash guard and numerical checks are the regression
oracle and stay untouched.  This module reuses that same qualified
pipeline for arbitrary mono input: waveform chunks are planned at the
mel frontend's 30-second window, resampling (when the source is not
16 kHz) and every tensor stage run on the accelerator path — MLX GPU
mel/decoder/joint, ANE islands or whole bundle for the encoder — with
the TDT predictor's recurrent state carried across chunk boundaries.

There is no CPU tensor path and no transcript is ever synthesized:
every prerequisite is verified by the same checks the CLI runs, and a
failure is a named refusal.  A hardware acceptance run records a
receipt (``write_acceptance_receipt``); only that receipt qualifies
arbitrary recognition, preflight never does.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import math
import os
import shutil
import struct
import sys
import threading
import time
from importlib.util import spec_from_loader
from pathlib import Path

SAMPLE_RATE = 16_000

_CHUNK_SAMPLES = 480_000
_RECEIPT_SCHEMA = "mlx-omarchy.assistant-recognition-acceptance.v1"


class DictationRefusal(RuntimeError):
    """The installed runtime cannot or must not run; the reason is named."""


class DictationCancelled(RuntimeError):
    """The caller's cancellation event fired during transcription."""


class DictationTimedOut(RuntimeError):
    """The transcription exceeded its deadline."""


def _installed_cli_candidates():
    """The mlx-omarchy install venv, independent of PATH and wrappers."""
    venv_roots = [Path.home() / ".local" / "share" / "mlx-omarchy" / "venv"]
    env_venv = os.environ.get("MLX_OMARCHY_VENV", "")
    if env_venv:
        venv_roots.insert(0, Path(env_venv))
    for root in venv_roots:
        yield from root.glob(
            "lib/python3.*/site-packages/mlx/bin/mlx-omarchy-parakeet"
        )


def _script_candidates():
    env = os.environ.get("MLX_OMARCHY_PARAKEET_TOOL", "")
    if env:
        yield Path(env)
    here = Path(__file__).resolve().parent
    yield here.parent / "mlx-omarchy-parakeet" / "mlx_omarchy_parakeet.py"
    yield from _installed_cli_candidates()
    found = shutil.which("mlx-omarchy-parakeet")
    if found:
        yield Path(found)


def _load_script_module(name: str, path: Path):
    """Import a Python file whatever its filename (installed CLI has none)."""
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = spec_from_loader(name, loader, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)  # the CLI re-derives its tree from __file__
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


def load_cli_module():
    """Import the pinned-product module from wherever it is installed."""
    for candidate in _script_candidates():
        if candidate.is_file():
            return _load_script_module("mlx_omarchy_parakeet", candidate)
    raise DictationRefusal(
        "the mlx-omarchy-parakeet CLI was not found (looked at "
        "MLX_OMARCHY_PARAKEET_TOOL, the mlx-omarchy-parakeet/ tree next to "
        "coreml/, the mlx-omarchy venv, and PATH); install the "
        "mlx-omarchy-parakeet product"
    )


def probe_runtime(*, verify_cache: bool = False) -> dict:
    """Report named, individually probed runtime facts. Never raises."""
    reasons: list[str] = []
    facts: dict[str, object] = {}

    try:
        cli = load_cli_module()
        facts["cli"] = str(Path(cli.__file__).resolve())
    except DictationRefusal as error:
        reasons.append(str(error))
        return {"ok": False, "reasons": reasons, "facts": facts}

    checks = (
        ("deps", lambda: cli._check_runtime_deps()),
        ("platform", lambda: cli._check_ane_capability()),
        ("assets", lambda: cli._verify_assets(cli._load_pin())),
        ("worker", lambda: cli._worker_path()),
    )
    for name, check in checks:
        try:
            check()
        except Exception as error:
            reasons.append(f"{name}: {error}")
        else:
            facts[name] = "ok"

    try:
        lock = cli.ReferenceLock.load()
        cache_dir = cli._cache_dir(lock)
        if not cache_dir.is_dir():
            raise DictationRefusal(
                f"the pinned reference cache is not downloaded "
                f"({cache_dir}); run `mlx-omarchy-parakeet download`"
            )
        facts["cache"] = "present"
        if verify_cache:
            ok, mismatches = cli.fetch.verify_cache(cache_dir, lock)
            if not ok:
                raise DictationRefusal(
                    f"the pinned reference cache does not verify "
                    f"({len(mismatches)} mismatched/missing files in "
                    f"{cache_dir}); run `mlx-omarchy-parakeet download`"
                )
            facts["cache"] = "verified"
    except Exception as error:
        facts["cache"] = "missing"
        reasons.append(f"cache: {error}")

    facts["sample_rate"] = SAMPLE_RATE
    facts["chunk_samples"] = _CHUNK_SAMPLES
    return {"ok": not reasons, "reasons": reasons, "facts": facts}


def plan_chunks(sample_count: int, chunk_samples: int = _CHUNK_SAMPLES) -> list[int]:
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    if chunk_samples <= 0:
        raise ValueError(f"chunk_samples must be positive, got {chunk_samples}")
    full, rest = divmod(sample_count, chunk_samples)
    plan = [chunk_samples] * full
    if rest:
        plan.append(rest)
    return plan


# ------------------------------------------------------------------
# Resampling on the accelerator path. Filter taps are host-designed
# constants uploaded to the device; all sample arithmetic is MLX.
# ------------------------------------------------------------------

def _kaiser_lowpass(up: int, down: int, half_len: int = 16) -> "object":
    """Unity-DC-gain FIR taps in up-sampled-sample units."""
    import numpy as np

    max_rate = max(up, down)
    half = half_len * up
    n = np.arange(-half, half + 1, dtype=np.float64)
    taps = np.sinc(n / max_rate) * np.kaiser(n.size, 8.0)
    return taps / taps.sum()


def resample_to_16k(waveform, sample_rate: int):
    """Resample mono float32 waveform to 16 kHz on the MLX device."""
    import numpy as np

    signal = np.ascontiguousarray(waveform, dtype=np.float32)
    if sample_rate == SAMPLE_RATE:
        return signal
    divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
    up = SAMPLE_RATE // divisor
    down = int(sample_rate) // divisor

    import mlx.core as mx

    taps = _kaiser_lowpass(up, down).astype(np.float32)
    rows = -(-taps.size // up)
    padded_taps = np.zeros(rows * up, dtype=np.float32)
    padded_taps[: taps.size] = taps
    half = (taps.size - 1) // 2

    device_taps = mx.array(padded_taps)
    device_signal = mx.pad(mx.array(signal), [(rows, rows)])
    n_out = -(-signal.size * up // down)
    m = mx.arange(n_out) * down + half
    phase = m % up
    base = m // up
    offsets = mx.arange(rows)
    gather_signal = mx.take(
        device_signal, (base[:, None] - offsets[None, :] + rows).reshape(-1)
    ).reshape(n_out, rows)
    gather_taps = mx.take(
        device_taps, (offsets[None, :] * up + phase[:, None]).reshape(-1)
    ).reshape(n_out, rows)
    resampled = (gather_signal * gather_taps).sum(axis=1)
    return np.asarray(resampled, dtype=np.float32)


# ------------------------------------------------------------------
# Acceptance receipts: preflight probes never qualify recognition; a
# recorded hardware acceptance run does.
# ------------------------------------------------------------------

def write_acceptance_receipt(home: Path, *, transcript: str,
                             emissions: int, host: str) -> Path:
    """Record a hardware acceptance run; callers pass the oracle checks."""
    cli = load_cli_module()
    pin = cli._load_pin()
    lock = cli.ReferenceLock.load()
    provenance = pin.get("provenance", {})
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "model_repo": provenance.get("model_repository"),
        "model_revision": lock.model_revision,
        "transcript_sha256": pin["e2e"]["transcript_sha256"],
        "transcript": transcript,
        "emissions": emissions,
        "host": host,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt_dir = Path(home) / "recognition"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    path = receipt_dir / "acceptance.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def read_acceptance_receipt(home: Path) -> dict | None:
    """The stored receipt when it matches the pinned revision, else None."""
    path = Path(home) / "recognition" / "acceptance.json"
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if receipt.get("schema") != _RECEIPT_SCHEMA:
        return None
    try:
        cli = load_cli_module()
        if receipt.get("model_revision") != cli.ReferenceLock.load().model_revision:
            return None
    except Exception:
        return None
    return receipt


class DictationTranscriber:
    """Long-lived arbitrary-audio transcriber over the pinned runtime."""

    def __init__(self, *, deadline_ms: int = 90_000):
        self._default_deadline_ms = deadline_ms
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self._island = None
        self._scratch = None
        self._runner = None
        self._packed = None
        self._tokenizer = None
        self._reference = None

    def _start(self):
        if self._started:
            return
        cli = load_cli_module()
        os.environ.setdefault("ANE_ISLAND_MODE", "resident-batch")
        try:
            pin = cli._load_pin()
            cli._check_runtime_deps()
            cli._check_ane_capability()
            cli._verify_assets(pin)
            worker = cli._worker_path()
            share = cli._share_dir()
        except (cli.TranscribeRefusal, cli.ReferenceError) as error:
            raise DictationRefusal(str(error)) from error

        try:
            lock = cli.ReferenceLock.load()
            cache_dir = cli._cache_dir(lock)
            ok, mismatches = cli.fetch.verify_cache(cache_dir, lock)
            if not ok:
                raise DictationRefusal(
                    "the reference cache does not verify against the lock; "
                    "run `mlx-omarchy-parakeet download` first "
                    f"({len(mismatches)} mismatched/missing files in "
                    f"{cache_dir})"
                )
            source = cli._ensure_encoder_source(lock, pin, cache_dir)
        except (cli.TranscribeRefusal, cli.ReferenceError) as error:
            raise DictationRefusal(str(error)) from error

        import tempfile

        import mlx.core as mx
        import numpy as np

        mx.set_default_device(mx.gpu)
        scratch = Path(tempfile.mkdtemp(
            prefix="parakeet-dictation-", dir=cli.default_cache_root()
        ))
        from coreml.parakeet_tdt import DecoderStep, JointDecision, tdt_decode
        from coreml.tokenizer import ParakeetTokenizer
        from coreml.vulkan_decoder import load_decoder
        from coreml.vulkan_decoder_step import pack_step_weights, run_step
        from coreml.vulkan_encoder import AneIsland, EncoderRunner
        from coreml.vulkan_mel import extract_chunk_features

        island = AneIsland(
            worker, share / "libane" / "libane-strict.so",
            share / "bundles", scratch, self._deadline_ms_value(),
        )
        self._island = island
        self._runner = EncoderRunner(
            source / "model.mil", source / "model-root", island,
        )
        decoder = load_decoder(cache_dir / "decoder.mlpackage")
        self._packed = pack_step_weights(decoder, cache_dir / "joint.mlpackage")
        tokenizer_sha = next(
            item.sha256 for item in lock.files
            if item.path == "tokenizer.json"
        )
        self._tokenizer = ParakeetTokenizer.load(
            cache_dir / "tokenizer.json", expected_sha256=tokenizer_sha
        )
        self._reference = lock
        self._np = np
        self._mx = mx
        self._extract_chunk_features = extract_chunk_features
        self._DecoderStep = DecoderStep
        self._JointDecision = JointDecision
        self._run_step = run_step
        self._tdt_decode = tdt_decode
        self._scratch = scratch
        self._started = True

    def _deadline_ms_value(self) -> int:
        return int(self._default_deadline_ms)

    @staticmethod
    def _cancelled(cancel) -> bool:
        return cancel is not None and cancel.is_set()

    def transcribe_waveform(self, waveform, sample_rate: int, cancel=None,
                            deadline_ms: int | None = None) -> str:
        """Transcribe mono float32 waveform at ``sample_rate``."""
        if self._closed:
            raise DictationRefusal("the transcriber has been closed")
        if self._cancelled(cancel):
            raise DictationCancelled("cancelled before transcription started")
        import numpy as np

        if not isinstance(waveform, np.ndarray) or waveform.dtype != np.float32:
            raise ValueError("waveform must be a numpy float32 array")
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("waveform must be a non-empty one-dimensional array")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"sample rate must be {SAMPLE_RATE} Hz, got {sample_rate}"
            )
        deadline = self._deadline_ms_value() if deadline_ms is None else deadline_ms
        started = time.monotonic()

        with self._lock:
            self._start()
            mx = self._mx
            np = self._np
            tokens: list[int] = []
            hidden = None
            cell = None
            samples = np.ascontiguousarray(waveform)
            for length in plan_chunks(int(samples.size)):
                if self._cancelled(cancel):
                    raise DictationCancelled("cancelled at a chunk boundary")
                if time.monotonic() - started > deadline / 1000.0:
                    raise DictationTimedOut(
                        f"transcription exceeded its {deadline} ms deadline"
                    )
                chunk = samples[:length]
                samples = samples[length:]
                chunk_tokens, hidden, cell = self._transcribe_chunk(
                    chunk, hidden, cell, cancel
                )
                tokens.extend(chunk_tokens)
            return self._tokenizer.decode(tokens)

    def _transcribe_chunk(self, chunk_waveform, hidden, cell, cancel):
        mx = self._mx
        with mx.stream(mx.gpu):
            waveform = mx.array(chunk_waveform)
        mx.eval(waveform)

        mel = self._extract_chunk_features(waveform)
        mx.eval(mel.mel, mel.mask, mel.encoder_features, mel.encoder_mask)

        encoded = self._runner.run(
            inputs={
                "input_features": mel.encoder_features,
                "attention_mask": mel.encoder_mask,
            },
            wanted={"encoder_hidden", "encoder_mask"},
            stop_after="encoder_mask",
        )
        encoder_hidden = encoded["encoder_hidden"].astype(mx.float32)
        encoder_mask = encoded["encoder_mask"].astype(mx.int32)
        mx.eval(encoder_hidden, encoder_mask)

        if hidden is None:
            with mx.stream(mx.gpu):
                hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
                cell = mx.zeros((2, 1, 640), dtype=mx.float32)
            mx.eval(hidden, cell)

        self._carry_shape(hidden)
        self._carry_shape(cell)

        fused = {"frame": None, "state": None, "tok": None, "dur": None}
        frame_holder = [0]

        def decoder_callback(token_id, current_hidden, current_cell):
            if self._cancelled(cancel):
                raise DictationCancelled("cancelled during decode")
            state_out, tok, dur, _, _, _ = self._run_step(
                self._packed, current_hidden, current_cell, token_id,
                encoder_hidden, frame_holder[0],
            )
            mx.eval(state_out)
            dec_state = state_out[0:640].reshape(1, 640)
            fused["frame"] = frame_holder[0]
            fused["state"] = dec_state
            fused["tok"] = tok
            fused["dur"] = dur
            return self._DecoderStep(
                dec_state,
                state_out[640:1920].reshape(2, 1, 640),
                state_out[1920:3200].reshape(2, 1, 640),
            )

        def joint_callback(frame_index, decoder_state):
            if self._cancelled(cancel):
                raise DictationCancelled("cancelled during decode")
            frame_holder[0] = frame_index
            if fused["frame"] == frame_index and fused["state"] is decoder_state:
                return self._JointDecision(fused["tok"], fused["dur"])
            state_out, tok, dur, _, _, _ = self._run_step(
                self._packed, None, None, 0, encoder_hidden, frame_index,
                skip_lstm=True, dec_in=decoder_state,
            )
            mx.eval(state_out)
            return self._JointDecision(tok, dur)

        tdt = self._tdt_decode(
            packed=self._packed,
            encoder=encoder_hidden,
            valid_frames=int(encoder_hidden.shape[1]),
            config=self._reference.tdt,
            initial_hidden=hidden,
            initial_cell=cell,
            run_decoder=decoder_callback,
            run_joint=joint_callback,
        )
        return list(tdt.token_ids), tdt.hidden, tdt.cell

    def _carry_shape(self, state) -> None:
        if state is None:
            return
        shape = tuple(state.shape)
        if shape != (2, 1, 640):
            raise DictationRefusal(
                "the decoder returned an unexpected recurrent state shape "
                f"{shape}; refusing to carry it across chunks"
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            island, self._island = self._island, None
            self._runner = None
            self._packed = None
            self._tokenizer = None
            if island is not None:
                try:
                    island.close()
                except Exception:
                    pass
            scratch, self._scratch = self._scratch, None
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)


# ------------------------------------------------------------------
# Owned worker process. The HTTP layer drives transcription through
# this protocol so a cancel or shutdown can always bound the work:
# the parent confirms child exit before releasing resources.
# ------------------------------------------------------------------

def _read_frame(stream) -> tuple[dict, bytes] | None:
    raw_header = stream.read(8)
    if len(raw_header) < 8:
        return None
    (header_size,) = struct.unpack("<Q", raw_header)
    header = json.loads(stream.read(header_size).decode("utf-8"))
    payload_size = int(header.get("payload_bytes", 0))
    payload = stream.read(payload_size) if payload_size else b""
    return header, payload


def _write_frame(stream, header: dict, payload: bytes = b"") -> None:
    header = dict(header)
    header["payload_bytes"] = len(payload)
    encoded = json.dumps(header).encode("utf-8")
    stream.write(struct.pack("<Q", len(encoded)) + encoded + payload)
    stream.flush()


def _worker_loop() -> int:
    transcriber: DictationTranscriber | None = None
    cancel_event = threading.Event()
    while True:
        try:
            frame = _read_frame(sys.stdin.buffer)
        except (struct.error, ValueError, json.JSONDecodeError):
            return 2
        if frame is None:
            break
        header, payload = frame
        op = header.get("op")
        if op == "close":
            _write_frame(sys.stdout.buffer, {"id": header.get("id"), "ok": True})
            break
        if op == "cancel":
            cancel_event.set()
            _write_frame(sys.stdout.buffer, {"id": header.get("id"), "ok": True})
            continue
        if op != "transcribe":
            _write_frame(sys.stdout.buffer, {
                "id": header.get("id"), "ok": False,
                "kind": "input", "error": f"unknown op {op!r}",
            })
            continue
        import numpy as np

        waveform = np.frombuffer(payload, dtype="<f4")
        if transcriber is None:
            transcriber = DictationTranscriber()
        cancel_event.clear()
        source_rate = int(header.get("sample_rate", SAMPLE_RATE))
        try:
            if source_rate != SAMPLE_RATE:
                waveform = resample_to_16k(waveform, source_rate)
            transcript = transcriber.transcribe_waveform(
                waveform, SAMPLE_RATE, cancel_event,
                deadline_ms=int(header.get("deadline_ms", 90_000)),
            )
        except DictationCancelled as error:
            _write_frame(sys.stdout.buffer, {
                "id": header.get("id"), "ok": False,
                "kind": "cancelled", "error": str(error),
            })
        except DictationTimedOut as error:
            _write_frame(sys.stdout.buffer, {
                "id": header.get("id"), "ok": False,
                "kind": "timeout", "error": str(error),
            })
        except (DictationRefusal, ImportError, ValueError) as error:
            _write_frame(sys.stdout.buffer, {
                "id": header.get("id"), "ok": False,
                "kind": "unavailable" if isinstance(error, (DictationRefusal, ImportError))
                else "input",
                "error": str(error),
            })
        else:
            _write_frame(sys.stdout.buffer, {
                "id": header.get("id"), "ok": True, "transcript": transcript,
            })
    if transcriber is not None:
        transcriber.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m coreml.parakeet_dictation")
    parser.add_argument("--worker", action="store_true",
                        help="serve the owned-worker framing protocol on stdio")
    args = parser.parse_args(argv)
    if not args.worker:
        parser.error("--worker is required")
    return _worker_loop()


if __name__ == "__main__":
    raise SystemExit(main())
