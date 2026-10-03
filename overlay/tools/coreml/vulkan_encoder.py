#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Run the pinned Parakeet encoder with the attention matmuls and the
attention-mask select on the ANE, and every other tensor op on the Apple GPU
through mlx-omarchy (Vulkan).

Phase 6 of docs/plans/2026-09-12-coreml-parakeet-ane-plan.md. Per transformer
layer, three ANE islands carry the attention:

  island A (bundle parakeet-encoder-island-attn-a-kt, 2 programs, 416 TDs)
      attention_scores_N = matmul(q_v, pos_kT)
      matmul_N           = matmul(q_scaled, k_headsT)
  island B (bundle parakeet-encoder-island-select-8head, 1 program, 5 TDs)
      attention_mask_N   = select(a = -inf fill, b = matrix_bd_N, cond = mask)
  island C (bundle parakeet-encoder-island-pv, 1 program, 208 TDs)
      attn_output_N      = matmul(probs, v_heads)

Everything else -- the subsampling conv stack, mask derivation, all layer
norms, both feed-forwards, the depthwise conv module, softmax, the 24
all-masked-rows selects, and the epilogue projector -- executes on ``mx.gpu``.

Island B is here because mil-hwx-compiler feature/h13-concat 7ab3eb5 fixed the
defect that kept it on the GPU in the two-island run: the H13 select program
under-declared its channel-3 scratch arena by 3x, so the cond-true half was
written and read past the declared surface. The rebuilt bundle declares 417
tiles / 6832128 bytes and returned 0 wrong lanes on m1-test-host. ``--islands`` selects
which islands are placed, so the two-island arm is reproducible from this same
script and the delta is attributable to placement rather than to a script edit.

This run does NOT establish that the ANE select is correct in general: the
golden clip's cond is uniformly false, so the -inf fill is never selected on
either device. See the receipt's coverage-gap section.

Section 43 (no CPU tensor fallback): every op is implemented with mlx.core
only and the runner raises on any op it cannot express in mx, so there is no
CPU arithmetic path to fall back to. Pinned constants are byte-reinterpreted
from the adapter's blob files straight into device arrays -- a load, not a
computation. Moving island tensors to and from the bounded ANE worker is
process I/O, also not arithmetic; both are accounted explicitly.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import hashlib
import importlib.metadata
import json
import os
import sys
import re
import struct
import subprocess
import time
from functools import cache
from pathlib import Path

import mlx.core as mx
import numpy as np

BLOB_MAGIC = 0xDEADBEEF

# Kill-switch for the fused chain+bias(+silu) epilogue: byte-identical when
# on, and off reproduces the stock dispatch stream exactly.
CHAIN_FUSION_ENABLED = (
    os.environ.get("MLX_OMARCHY_CHAIN_FUSION", "1") not in ("0", "false", "no")
)

# Device-resident const cache: materialize every const once per runner, keep
# the device buffers referenced, and re-reference them (dict insert, no
# upload) on later passes. Opt-in (default off): holding ~1.15 GB resident
# costs a fresh-process pass ~+236 ms (the allocator cannot recycle the held
# buffers for intermediates), so single-pass pipelines -- including the E2E,
# which calls run() exactly once per process -- run unchanged by default;
# multi-pass consumers set MLX_OMARCHY_ENCODER_CONST_CACHE=1 and save
# ~1.05-1.13 s per warm pass.
CONST_CACHE_ENABLED = (
    os.environ.get("MLX_OMARCHY_ENCODER_CONST_CACHE", "0") not in ("0", "false", "no")
)

STMT = re.compile(
    r"^\s*(?P<type>tensor<[^>]*>|string|int32|bool|fp16|fp32)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_@]*)\s*=\s*"
    r"(?P<op>[a-z_][a-z_0-9]*)\((?P<args>.*)\)\s*(?:\[(?P<attrs>.*)\])?;\s*$"
)
TUPLE_STMT = re.compile(
    r"^\s*\((?P<results>tensor<[^)]*)\)\s*=\s*"
    r"(?P<op>[a-z_][a-z_0-9]*)\((?P<args>.*)\)\s*(?:\[(?P<attrs>.*)\])?;\s*$"
)
TYPE = re.compile(r"^tensor<\s*(?P<dtype>\w+)\s*,\s*\[(?P<shape>[^\]]*)\]>$")
BLOBFILE = re.compile(
    r'BLOBFILE\(path = string\("(?P<path>[^"]+)"\), offset = uint64\((?P<offset>\d+)\)\)'
)

MX_DTYPES = {"fp16": mx.float16, "fp32": mx.float32, "int32": mx.int32, "bool": mx.bool_}
NP_DTYPES = {"fp16": np.float16, "fp32": np.float32, "int32": np.int32, "bool": np.bool_}

# Identify the spliced islands by the MIL result names they produce.
RE_SCORES = re.compile(r"^attention_scores_\d+_cast_fp16$")
RE_CONTENT = re.compile(r"^matmul_\d+_cast_fp16$")
RE_ATTN_OUT = re.compile(r"^attn_output_\d+_cast_fp16$")
# Island B. Two traps here. The MIL also carries 24 all-masked-rows selects
# named input_N, so the name match excludes them and the shape assertion in
# _index_islands is the second gate. And the 24th mask select is
# attention_mask_cast_fp16, with no ordinal at all -- Core ML drops the suffix
# on the last of a repeated family -- so the ordinal is optional. A \d+-only
# pattern silently finds 23 of 24 and the balance check is what catches it.
RE_MASK_SELECT = re.compile(r"^attention_mask(_\d+)?_cast_fp16$")
ISLAND_B_SHAPE = (1, 8, 375, 375)



# Metal kernel factories live in vulkan_kernels (issue #17 split);
# re-exported here so every handler keeps resolving the same names.
from coreml.vulkan_kernels import (
    _leftover_chain_kernel,
    _leftover_chain_bias_kernel,
    _leftover_chain_bias_silu_kernel,
    _linear_f16_coopmat_kernel,
    _silu_kernel,
    _glu_kernel,
    _ln_cast_kernel,
    _ln_sq_kernel,
    _ln_tail_kernel,
    _sm_exp_kernel,
    _sm_div_kernel,
    _pw_kernel,
    _pw,
    _sigmoid_kernel,
)
class EncoderRunError(RuntimeError):
    """The run cannot continue; the reason is named."""


class _TraceSnapshot(ctypes.Structure):
    _fields_ = [
        ("gpu_primitive_dispatches", ctypes.c_uint64),
        ("vk_submissions", ctypes.c_uint64),
        ("vk_buffer_copies", ctypes.c_uint64),
        ("vk_buffer_fills", ctypes.c_uint64),
        ("vk_compute_dispatches", ctypes.c_uint64),
        ("omarchy_finalize_calls", ctypes.c_uint64),
        ("commit_calls_with_work", ctypes.c_uint64),
        ("commit_calls_noop", ctypes.c_uint64),
    ]


@cache
def _trace_function():
    distribution = importlib.metadata.distribution("mlx-omarchy")
    library_path = distribution.locate_file("mlx/lib/libmlx.so")
    library = ctypes.CDLL(str(library_path))
    function = library.mlx_omarchy_trace_snapshot
    function.argtypes = [ctypes.POINTER(_TraceSnapshot)]
    function.restype = None
    return function


def vm_rss_kb() -> int:
    """Resident set size in KiB from /proc, for leak-watch evidence."""
    for line in Path("/proc/self/status").read_text().split("\n"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


def trace_snapshot() -> dict[str, int]:
    snapshot = _TraceSnapshot()
    _trace_function()(ctypes.byref(snapshot))
    return {name: int(getattr(snapshot, name)) for name, _ in snapshot._fields_}


def split_top(text: str) -> list[str]:
    out, depth, start, quoted = [], 0, 0, False
    for i, ch in enumerate(text):
        if quoted:
            if ch == '"':
                quoted = False
            continue
        if ch == '"':
            quoted = True
        elif ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(text[start:i])
            start = i + 1
    tail = text[start:]
    if tail.strip():
        out.append(tail)
    return [item.strip() for item in out]


def parse_kwargs(text: str) -> dict[str, str]:
    result = {}
    for item in split_top(text):
        if not item:
            continue
        key, _, value = item.partition("=")
        result[key.strip()] = value.strip()
    return result


def parse_type(text: str):
    match = TYPE.match(text)
    if match:
        name = match.group("dtype").lower()
        shape_text = match.group("shape").strip()
        shape = tuple(int(i) for i in shape_text.split(",")) if shape_text else ()
        return name, shape
    return text.strip().lower(), ()


class Blobs:
    """Reader for the blob-v2 files overlay/tools/coreml/mil_adapter.py emits."""

    def __init__(self, model_root: Path):
        self.root = model_root
        self._maps: dict[str, np.memmap] = {}

    def _map(self, path: str) -> np.memmap:
        name = path.replace("@model_path/", "")
        if name not in self._maps:
            self._maps[name] = np.memmap(self.root / name, dtype=np.uint8, mode="r")
        return self._maps[name]

    def read_bytes(self, path: str, offset: int, want: int) -> memoryview:
        raw = self._map(path)
        magic, _storage, length, payload = struct.unpack_from(
            "<IIQQ", raw[offset : offset + 24].tobytes(), 0
        )
        if magic != BLOB_MAGIC:
            raise EncoderRunError(f"blob magic {magic:#x} at {path}:{offset}")
        if want > length:
            raise EncoderRunError(
                f"blob at {path}:{offset} holds {length} bytes, want {want}"
            )
        return memoryview(raw[payload : payload + want].tobytes())


class Statement:
    __slots__ = (
        "index", "names", "op", "kwargs", "attrs", "dtype", "shape", "done",
        "operands", "const_kwargs",
    )

    def __init__(self, index, names, op, kwargs, attrs, dtype, shape):
        self.index = index
        self.names = names
        self.op = op
        self.kwargs = kwargs
        self.attrs = attrs
        self.dtype = dtype
        self.shape = shape
        self.done = False
        self.operands: list[str] = []
        self.const_kwargs: dict | None = None


# Island bundles the encoder handlers may submit to. The resident session
# preloads exactly the registered sites: the static defaults below, extended
# at index time with every bundle the placed set can submit (the oproj
# family registers island-oproj-L* per layer).
RESIDENT_BUNDLES = (
    "island-attn-a-kt",
    "island-select-8head",
    "island-pv",
)

# The whole-program encoder: Apple's single-program Parakeet encoder
# container (hwxv2 converter, 512-B tile units) executed in ONE submit.
# programs[0].operation == "whole-encoder" in its manifest marks it. When a
# whole-program bundle is discoverable (explicit MLX_OMARCHY_WHOLE_ENCODER_BUNDLE
# directory, the bundles share, or a compiled-cache entry) the encoder runner
# sends the whole graph in one ANE submit; otherwise it falls back to the
# island path unchanged. MLX_OMARCHY_WHOLE_ENCODER=0 forces the islands.
WHOLE_ENCODER_OP = "whole-encoder"


def _manifest_operation(manifest_path: Path) -> str | None:
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    programs = manifest.get("programs") if isinstance(manifest, dict) else None
    if not isinstance(programs, list) or not programs:
        return None
    op = programs[0].get("operation") if isinstance(programs[0], dict) else None
    return op if isinstance(op, str) else None


def _pin_declares_whole_bundle() -> bool:
    """True iff the runtime pin read from the share tree names the
    whole-encoder bundle. Used to decide whether a missing bundle is a
    silent-fallback case or a deliberate opt-out (MLX_OMARCHY_WHOLE_ENCODER=0).
    """
    pin_candidates = [
        # installed wheel layout
        Path(__file__).resolve().parents[1] / "share" / "mlx-omarchy" /
        "parakeet-1" / "parakeet-runtime-pin.json",
        # source-tree layout
        Path(__file__).resolve().parents[2] / "mlx-omarchy-parakeet" /
        "share" / "mlx-omarchy" / "parakeet-1" / "parakeet-runtime-pin.json",
    ]
    for pin_path in pin_candidates:
        if not pin_path.is_file():
            continue
        try:
            pin = json.loads(pin_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        bundles = pin.get("assets", {}).get("bundles", {}) if isinstance(pin, dict) else {}
        if isinstance(bundles, dict) and "parakeet-encoder-whole" in bundles:
            return True
    return False


def _discover_whole_bundle() -> Path | None:
    """Locate the whole-program encoder bundle, cheapest source first.

    The worker re-verifies every payload digest against its manifest on
    every open, so discovery only names a directory; integrity stays with
    the loader and the compiled cache's stored digests.
    """
    override = os.environ.get("MLX_OMARCHY_WHOLE_ENCODER_BUNDLE", "")
    if override:
        path = Path(override)
        if _manifest_operation(path / "manifest.json") == WHOLE_ENCODER_OP:
            return path
        raise EncoderRunError(
            f"MLX_OMARCHY_WHOLE_ENCODER_BUNDLE {override} is not a "
            f"whole-encoder bundle directory"
        )
    roots = [
        Path(__file__).resolve().parents[2] / "mlx-omarchy-parakeet" / "share" /
        "mlx-omarchy" / "parakeet-1" / "bundles",
        # installed wheel layout: site-packages/mlx/coreml/vulkan_encoder.py
        # -> site-packages/mlx/share/mlx-omarchy/parakeet-1/bundles
        Path(__file__).resolve().parents[1] / "share" / "mlx-omarchy" /
        "parakeet-1" / "bundles",
    ]
    try:
        from coreml.compiled_cache import default_cache_root

        roots.append(default_cache_root())
    except Exception:
        pass
    for root in roots:
        if not root.is_dir():
            continue
        for manifest_path in sorted(root.rglob("manifest.json")):
            if _manifest_operation(manifest_path) == WHOLE_ENCODER_OP:
                return manifest_path.parent
    # Fail loud: when the runtime pin declares the whole-encoder bundle
    # as a required asset, a missing discovery is no longer a quiet
    # fallback to the split-island path -- the installed wheel was
    # shipped incomplete. Force the operator to either supply the bundle
    # or set MLX_OMARCHY_WHOLE_ENCODER=0 to opt out of the whole path
    # deliberately. A nil discovery with no opt-out here would silently
    # take the 705-1050 ms split-island path, which is the original bug.
    if _pin_declares_whole_bundle() and os.environ.get(
        "MLX_OMARCHY_WHOLE_ENCODER", "1"
    ).strip().lower() not in ("", "0", "off", "false", "no"):
        searched = ", ".join(str(r) for r in roots if r.is_dir()) or "(no share root found)"
        raise EncoderRunError(
            "the runtime pin declares parakeet-encoder-whole but the bundle "
            "directory was not discoverable; refusing to silently fall back "
            "to the split-island encoder path. Searched: " + searched +
            ". Fix: install the bundle with scripts/install-integrated.sh, "
            "set MLX_OMARCHY_WHOLE_ENCODER_BUNDLE=/path/to/dir, or set "
            "MLX_OMARCHY_WHOLE_ENCODER=0 to opt out of the whole path."
        )
    return None


# Process-level resident session singleton: repeated transcriptions in one
# process reuse the spawned worker and its resident bundles instead of paying
# worker spawn + bundle register + device program load on every pass. The
# held session is keyed on the full session identity (worker, libane, bundle
# name/path set, deadlines), so a different identity simply spawns its own
# session. ANE_ISLAND_PRIVATE_SESSION=1 (or true/yes/on) opts out and gives
# every AneIsland its own private session, the pre-singleton behavior. The
# worker is a private child process either way; the atexit hook releases it
# when the process exits, outside any measured pass.
_SHARED_SESSION: dict | None = None
_SHARED_SESSION_ATEXIT = False


def _shared_session_take(identity: tuple):
    """Return the held session when identity matches and it is still up.

    An identity change (worker, library, bundle set, deadlines, or the
    load-boundary pins) retires the held session BEFORE the caller
    spawns its replacement: the old child is a private worker holding
    the device and its sealed images, and overwriting the registry slot
    would orphan it. The caller only proceeds once the old child is
    confirmed reaped.
    """
    global _SHARED_SESSION
    entry = _SHARED_SESSION
    if entry is None:
        return None
    if entry["identity"] == identity and entry["session"].alive:
        return entry["session"]
    _retire_shared_session(entry["session"])
    _SHARED_SESSION = None
    return None


def _retire_shared_session(session) -> None:
    """Close a held session and confirm its child was actually reaped."""
    # Retain the Popen before close(): close() releases the handle, and
    # the kernel's own reap status (wait/poll) is the only authoritative
    # evidence — a recorded pid could be recycled, so it proves nothing.
    process = session._process
    try:
        if session.alive:
            session.close()
    finally:
        session._terminate()
    if process is not None and process.poll() is None:
        raise RuntimeError(
            "retired resident worker child has not exited; "
            "refusing to open a replacement session"
        )


def _shared_session_hold(identity: tuple, session) -> None:
    global _SHARED_SESSION, _SHARED_SESSION_ATEXIT
    _SHARED_SESSION = {"identity": identity, "session": session}
    if not _SHARED_SESSION_ATEXIT:
        _SHARED_SESSION_ATEXIT = True
        atexit.register(_shared_session_release)


def _shared_session_drop(session) -> None:
    """Forget a session that died mid-pass so nothing reuses a corpse."""
    global _SHARED_SESSION
    if _SHARED_SESSION is not None and _SHARED_SESSION["session"] is session:
        _SHARED_SESSION = None


def _shared_session_release() -> None:
    """Close the held session at process exit; best effort by design."""
    global _SHARED_SESSION
    entry, _SHARED_SESSION = _SHARED_SESSION, None
    if entry is None:
        return
    session = entry["session"]
    try:
        if session.alive and session._batch_until is not None:
            session.end_batch()
    except Exception:
        pass
    try:
        if session.alive:
            session.close()
    except Exception:
        session._terminate()


class AneIsland:
    """One bounded submit to the physical ANE through mlx-omarchy-ane-worker.

    ANE_ISLAND_MODE picks the path:

    launch (default) -- one process per submit: the worker CLI takes a
    single bundle and a single input set, so a 24-layer encoder needs one
    launch per island per layer.

    resident-batch -- one private ``--serve`` worker for the whole pass and
    ONE batch scope: every layer's island submit is a round inside a single
    deadline-bounded batch (ANE_ISLAND_BATCH_DEADLINE_MS, default 120000).
    One process, one bundle load, one bounded unit for all 72 submits; a
    failed round or a deadline miss ends the session and is reported, never
    retried.
    """

    def __init__(self, worker: Path, libane: Path, bundles: Path, scratch: Path,
                 deadline_ms: int = 20000, seal_assets: dict | None = None):
        self.worker = worker
        self.libane = libane
        self.bundles = bundles
        self.scratch = scratch
        self.deadline_ms = deadline_ms
        # Approved asset digests in the runtime pin's "assets" shape
        # ("bundles": dir name -> file -> sha256, "libane": file ->
        # sha256). When set, the worker seals every consumed byte at
        # session open and refuses any mismatch before device load;
        # every session bundle and the library must be covered.
        self.seal_assets = seal_assets
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.submissions = 0
        self.worker_starts = 0
        self.rounds = 0
        self.input_bytes = 0
        self.output_bytes = 0
        self.exec_ns = 0
        # Host-side split around the timed submit: marshal_ns covers turning
        # input mx tensors into wire bytes (mx.eval + ascontiguous + tobytes),
        # back_ns covers turning output bytes into mx tensors (frombuffer +
        # mx.array). Everything ane_exec does not explain between the island
        # statement and its result lives in these two or in session open.
        self.marshal_ns = 0
        self.back_ns = 0
        self.timeouts = 0
        self.batch_open_ns = 0
        self.log: list[dict] = []
        # Default transport is the serve worker (spawn paid once): t6001-test-host
        # measured resident-batch beating launch by ~730-830 ms with all
        # pins EXACT - the per-submit spawn+init+bundle cost is not
        # hideable behind GPU feeder compute (data-dependent serial chain,
        # no independent GPU work during spawn windows).
        self._mode = os.environ.get("ANE_ISLAND_MODE", "resident-batch")
        if self._mode not in ("launch", "resident-batch"):
            raise EncoderRunError(
                f"ANE_ISLAND_MODE {self._mode!r} is not launch or resident-batch"
            )
        self._batch_deadline_ms = int(
            os.environ.get("ANE_ISLAND_BATCH_DEADLINE_MS", "120000")
        )
        self.resident_bundles = set(RESIDENT_BUNDLES)
        self._session = None
        # Session singleton bookkeeping: worker_starts counts actual spawns
        # this island performed (0 when it reused the shared session),
        # session_open_ns is the full spawn+register+load cost (0 on reuse),
        # batch_open_ns is this pass's ensure cost (spawn+register+batch on
        # first use, begin_batch alone on reuse), session_close_ns the
        # end_batch scope close this pass pays.
        self.share_session = os.environ.get(
            "ANE_ISLAND_PRIVATE_SESSION", ""
        ).strip().lower() not in ("1", "true", "yes", "on")
        self.session_reused = False
        # "daemon" when the session attached to a resident daemon
        # (MLX_OMARCHY_PK_KEEP_WORKER), "private" for a worker spawned
        # by this process; None until the first submit opens a session.
        self.session_transport = None
        self.session_open_ns = 0
        self.session_close_ns = 0
        self._session_shared = False
        # Whole-program encoder bundle, when discoverable. Kept as a
        # resolved directory so the launch path can pass it directly; the
        # resident session registers it by manifest name.
        self.whole_bundle: Path | None = None
        if os.environ.get("MLX_OMARCHY_WHOLE_ENCODER", "1").strip().lower() \
                not in ("", "0", "off", "false", "no"):
            self.whole_bundle = _discover_whole_bundle()

    def close(self) -> None:
        """End this pass's batch scope; a shared session stays resident."""
        if self._session is None:
            return
        session, self._session = self._session, None
        started = time.monotonic_ns()
        try:
            session.end_batch()
        except Exception as error:
            if self._session_shared:
                _shared_session_drop(session)
            raise EncoderRunError(f"resident batch close failed: {error}") from error
        self.session_close_ns = time.monotonic_ns() - started
        if not self._session_shared:
            try:
                session.close()
            except Exception as error:
                raise EncoderRunError(f"resident session close failed: {error}") from error

    def _session_bundles(self) -> dict:
        """The resident bundle set this session registers, name -> path."""
        bundles = {
            name: Path(self.bundles) / name
            for name in sorted(self.resident_bundles)
        }
        if self.whole_bundle is not None:
            manifest = json.loads(
                (self.whole_bundle / "manifest.json").read_text()
            )
            bundles[manifest["name"]] = Path(self.whole_bundle)
        return bundles

    def _seal_kwargs(self, bundles: dict) -> dict:
        """Per-session pin arguments derived from the approved assets.

        Every registered session bundle must be covered by the pin
        (looked up by bundle directory name) and the device library by
        its file name; an uncovered bundle or library is a refusal, so
        an installed-but-unpinned asset can never execute.
        """
        if self.seal_assets is None:
            return {}
        pinned_bundles = self.seal_assets.get("bundles", {})
        seal_expects: dict[str, dict[str, str]] = {}
        for name, path in bundles.items():
            files = pinned_bundles.get(Path(path).name)
            if not files:
                raise EncoderRunError(
                    f"resident bundle {name} ({Path(path).name}) is not "
                    "covered by the approved assets; refusing to load "
                    "unpinned ANE programs"
                )
            seal_expects[name] = {str(k): str(v) for k, v in files.items()}
        libane_files = self.seal_assets.get("libane", {})
        seal_libane_sha = libane_files.get(Path(self.libane).name)
        if not seal_libane_sha:
            raise EncoderRunError(
                f"device library {Path(self.libane).name} is not covered "
                "by the approved assets; refusing to load unpinned ANE "
                "userspace"
            )
        return {
            "seal_expects": seal_expects,
            "seal_libane_sha": str(seal_libane_sha),
        }

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ane_resident import ResidentAneWorker
        started = time.monotonic_ns()
        bundles = self._session_bundles()
        identity = (
            str(self.worker),
            str(self.libane),
            tuple(sorted((name, str(path)) for name, path in bundles.items())),
            self.deadline_ms,
            self._batch_deadline_ms,
            self._seal_kwargs(bundles),
        )
        session = None
        if self.share_session:
            session = _shared_session_take(identity)
            self._session_shared = True
            self.session_reused = session is not None
        if session is None:
            # With MLX_OMARCHY_PK_KEEP_WORKER=1 and MLX_OMARCHY_ANE_SOCK
            # set, start() attaches to the resident daemon and falls back
            # to a private worker on any connect failure.
            session = ResidentAneWorker(
                worker=Path(self.worker),
                libane=Path(self.libane),
                bundles=bundles,
                scratch=Path(self.scratch),
                deadline_ms=self.deadline_ms,
                **self._seal_kwargs(bundles),
            )
            session.start()
            if self.share_session:
                _shared_session_hold(identity, session)
                self._session_shared = True
        session.begin_batch(self._batch_deadline_ms)
        self.batch_open_ns = time.monotonic_ns() - started
        self.session_transport = session.transport
        if not self.session_reused:
            if session.transport == "private":
                self.worker_starts += 1
            self.session_open_ns = self.batch_open_ns
        self.submissions += 1
        self._session = session
        return session

    def _submit_resident(self, bundle: str, tag: str, inputs: dict, outputs: dict) -> dict:
        """One round inside the open batch: same bytes, no files, one session."""
        session = self._ensure_session()
        payload = {}
        in_bytes = 0
        marshal_started = time.monotonic_ns()
        ordered_inputs = list(inputs.items())
        # batch-eval: ONE graph walk + sync for all inputs instead of one
        # per tensor (measured: marshal segment is ~97% mx.eval readiness
        # wait; batching removes up to 144 redundant syncs per pass).
        mx.eval(*[value for _, value in ordered_inputs])
        for name, value in ordered_inputs:
            raw = np.ascontiguousarray(np.asarray(value)).tobytes()
            payload[name] = raw
            in_bytes += len(raw)
        round_marshal_ns = time.monotonic_ns() - marshal_started
        self.marshal_ns += round_marshal_ns
        out_names = list(outputs)
        started = time.monotonic_ns()
        try:
            results = session.submit(bundle, tag, payload, out_names)
        except Exception as error:
            if self._session_shared:
                _shared_session_drop(session)
            raise EncoderRunError(
                f"ANE batch round {tag} ({bundle}) failed: {error}"
            ) from error
        elapsed = time.monotonic_ns() - started
        back_started = time.monotonic_ns()
        self.rounds += 1
        self.exec_ns += elapsed
        self.input_bytes += in_bytes
        record = {"tag": tag, "bundle": bundle, "elapsed_ns": elapsed, "round": True}
        child = getattr(session, "log", [None])[-1] if session.log else None
        if child:
            for key in ("write_ns", "read_ns", "stage_ms", "save_ms", "elapsed_ms"):
                if key in child:
                    record[key] = child[key]
        out_bytes = 0
        packed = {}
        for name, (shape, dtype_name) in outputs.items():
            raw = results[name]
            count = 1
            for dim in shape:
                count *= dim
            expect = count * np.dtype(NP_DTYPES[dtype_name]).itemsize
            if len(raw) != expect:
                raise EncoderRunError(
                    f"ANE output {name} for {tag} is {len(raw)} bytes, want {expect}"
                )
            out_bytes += len(raw)
            host = np.frombuffer(raw, dtype=NP_DTYPES[dtype_name], count=count)
            packed[name] = mx.array(host).reshape(shape)
        round_back_ns = time.monotonic_ns() - back_started
        self.back_ns += round_back_ns
        record["back_ns"] = round_back_ns
        record["marshal_ns"] = round_marshal_ns
        record["input_bytes"] = in_bytes
        record["output_bytes"] = out_bytes
        self.output_bytes += out_bytes
        self.log.append(record)
        return packed

    def submit(self, bundle: str, tag: str, inputs: dict, outputs: dict) -> dict:
        """inputs: name -> mx array. outputs: name -> (shape, dtype name)."""
        if self._mode == "resident-batch":
            return self._submit_resident(bundle, tag, inputs, outputs)
        return self._submit_launch(self.bundles / bundle, tag, inputs, outputs)

    def submit_whole(self, tag: str, inputs: dict, outputs: dict) -> dict:
        """One whole-program submit.

        Resident mode rides the open batch under the bundle's manifest
        name; launch mode passes the resolved bundle directory directly.
        """
        manifest = json.loads((self.whole_bundle / "manifest.json").read_text())
        if self._mode == "resident-batch":
            return self._submit_resident(manifest["name"], tag, inputs, outputs)
        return self._submit_launch(Path(self.whole_bundle), tag, inputs, outputs)

    def _submit_launch(self, bundle, tag: str, inputs: dict, outputs: dict) -> dict:
        bundle = Path(bundle)
        run_dir = self.scratch / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            str(self.worker),
            "--bundle", str(bundle),
            "--libane", str(self.libane),
            "--deadline-ms", str(self.deadline_ms),
            "--iterations", "1",
        ]
        in_bytes = 0
        marshal_started = time.monotonic_ns()
        for name, value in inputs.items():
            mx.eval(value)
            raw = np.ascontiguousarray(np.asarray(value))
            path = run_dir / f"in_{name}.bin"
            path.write_bytes(raw.tobytes())
            in_bytes += raw.nbytes
            argv += ["--input", f"{name}={path}"]
        round_marshal_ns = time.monotonic_ns() - marshal_started
        self.marshal_ns += round_marshal_ns
        saved = {}
        for name in outputs:
            path = run_dir / f"out_{name}.bin"
            saved[name] = path
            argv += ["--save", f"{name}={path}"]

        self.worker_starts += 1
        started = time.monotonic_ns()
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=None)
        elapsed = time.monotonic_ns() - started
        self.submissions += 1
        self.exec_ns += elapsed
        self.input_bytes += in_bytes

        record = {
            "tag": tag,
            "bundle": bundle,
            "exit": proc.returncode,
            "elapsed_ns": elapsed,
            "stdout": proc.stdout.strip().splitlines(),
            "stderr": proc.stderr.strip().splitlines(),
        }
        self.log.append(record)
        if proc.returncode != 0:
            if "errno" in proc.stderr and "110" in proc.stderr:
                self.timeouts += 1
            raise EncoderRunError(
                f"ANE submit {tag} ({bundle}) exited {proc.returncode}: "
                f"{proc.stderr.strip()[:400]}"
            )

        results = {}
        out_bytes = 0
        back_started = time.monotonic_ns()
        for name, (shape, dtype_name) in outputs.items():
            raw = saved[name].read_bytes()
            count = 1
            for dim in shape:
                count *= dim
            expect = count * np.dtype(NP_DTYPES[dtype_name]).itemsize
            if len(raw) != expect:
                raise EncoderRunError(
                    f"ANE output {name} for {tag} is {len(raw)} bytes, want {expect}"
                )
            out_bytes += len(raw)
            host = np.frombuffer(raw, dtype=NP_DTYPES[dtype_name], count=count)
            results[name] = mx.array(host).reshape(shape)
            saved[name].unlink()
        round_back_ns = time.monotonic_ns() - back_started
        self.back_ns += round_back_ns
        record["marshal_ns"] = round_marshal_ns
        record["back_ns"] = round_back_ns
        self.output_bytes += out_bytes
        record["input_bytes"] = in_bytes
        record["output_bytes"] = out_bytes
        for path in run_dir.glob("in_*.bin"):
            path.unlink()
        return results


class EncoderRunner:
    def __init__(self, mil_path: Path, model_root: Path, island: AneIsland | None,
                 placed: frozenset[str] = frozenset(
                     os.environ.get("MLX_OMARCHY_PLACED", "AC"))):
        # Whole-program selection happens before any MIL machinery: when a
        # whole-encoder bundle is discoverable the MIL text is never parsed
        # and the entire graph is one ANE submit. Missing bundle -> island
        # fallback below, unchanged.
        self.whole_manifest: dict | None = None
        if island is not None and island.whole_bundle is not None:
            self.whole_manifest = json.loads(
                (island.whole_bundle / "manifest.json").read_text()
            )
        if self.whole_manifest is None:
            self.text = mil_path.read_text()
        else:
            self.text = ""
        self._init_island(mil_path, model_root, island, placed)

    def _init_island(self, mil_path: Path, model_root: Path,
                     island: AneIsland | None,
                     placed: frozenset[str]) -> None:
        self.blobs = Blobs(model_root)
        self.island = island
        self.placed = placed if island is not None else frozenset()
        self.values: dict[str, mx.array] = {}
        self.meta: dict[str, object] = {}
        self.statements: list[Statement] = []
        self.producer: dict[str, Statement] = {}
        self.const_stmt: dict[str, Statement] = {}
        self.const_values: dict[str, mx.array] = {}
        self.const_meta: dict[str, object] = {}
        self.executed = 0
        self.gpu_ops = 0
        self.ane_ops = 0
        # Wall-clock sum per statement op across the pass. Anything the island
        # timers (marshal/back/exec) do not explain shows up here, bucketed by
        # op name, so the encoder wall decomposes without a second run.
        self.op_wall_ns: dict[str, int] = {}
        self.cpu_tensor_events = 0
        self.cond_census: dict | None = None
        # MLX_OMARCHY_PIPE: issue-only async_eval per GPU statement so the
        # Vulkan queue stays saturated between island-boundary drains.
        # Scheduling only - same graph, same values, no host sync.
        # Default on (measured t6001-test-host: AC launch -1185ms, resident -813ms,
        # ACO launch -1547ms, resident -1099ms, all pins EXACT);
        # set MLX_OMARCHY_PIPE=0 to opt out.
        self.pipe = os.environ.get("MLX_OMARCHY_PIPE", "1") == "1"
        # Coarser issue cadence beats per-statement (t6001-test-host sweep: conv-only
        # 6561/5867 vs all-ops 7655/7179 AC launch/resident) - async_eval at
        # conv statements only.
        self.pipe_ops = frozenset(
            os.environ.get("MLX_OMARCHY_PIPE_OPS", "conv").split(",")
        ) - {""}
        self.glu_fusions: dict[int, tuple[str, str]] = {}
        self.glu_sigmoid_done: set[int] = set()
        self.linear_silu: dict[int, int] = {}
        self.silu_done: set[int] = set()
        self._parse()
        self._index_islands()
        self._index_fusions()
        self._last_use()
        if self.whole_manifest is not None:
            self.placed = frozenset("W")
            self.ane_ops = sum(
                program.get("task_descriptors", 0)
                for program in self.whole_manifest["programs"]
            )

    # ---------------------------------------------------------------- parsing

    def _parse(self) -> None:
        index = 0
        for line in self.text.split("\n"):
            tup = TUPLE_STMT.match(line)
            if tup is not None:
                names = [
                    item.rsplit(" ", 1)[1] for item in split_top(tup.group("results"))
                ]
                stmt = Statement(
                    index, names, tup.group("op"),
                    parse_kwargs(tup.group("args")), tup.group("attrs") or "", None, None,
                )
            else:
                match = STMT.match(line)
                if match is None:
                    continue
                dtype, shape = parse_type(match.group("type"))
                stmt = Statement(
                    index, [match.group("name")], match.group("op"),
                    parse_kwargs(match.group("args")), match.group("attrs") or "",
                    dtype, shape,
                )
            self.statements.append(stmt)
            if stmt.op == "const":
                stmt.const_kwargs = parse_kwargs(stmt.attrs)
            for name in stmt.names:
                self.producer[name] = stmt
                if stmt.op == "const":
                    self.const_stmt[name] = stmt
            index += 1
        if self.statements and self.statements[-1].index != len(self.statements) - 1:
            raise EncoderRunError("statement index sequence is corrupt")

    def _index_islands(self) -> None:
        scores, content, attn_out, mask_select = [], [], [], []
        for stmt in self.statements:
            name = stmt.names[0]
            if stmt.op == "matmul":
                if RE_SCORES.match(name):
                    scores.append(stmt)
                elif RE_CONTENT.match(name):
                    content.append(stmt)
                elif RE_ATTN_OUT.match(name):
                    attn_out.append(stmt)
            elif stmt.op == "select" and RE_MASK_SELECT.match(name):
                if tuple(stmt.shape) != ISLAND_B_SHAPE:
                    raise EncoderRunError(
                        f"{name} is {tuple(stmt.shape)}, island B bundle is "
                        f"{ISLAND_B_SHAPE}"
                    )
                mask_select.append(stmt)
        if not (len(scores) == len(content) == len(attn_out) == len(mask_select)):
            raise EncoderRunError(
                "island sets unbalanced: "
                f"{len(scores)}/{len(content)}/{len(mask_select)}/{len(attn_out)}"
            )
        self.layers = len(scores)
        # island A pairs the i-th rel-pos matmul with the i-th content matmul.
        self.island_a = {s.index: (i, s, c) for i, (s, c) in enumerate(zip(scores, content))}
        self.island_a_partner = {c.index: s.index for s, c in zip(scores, content)}
        self.island_b = {s.index: (i, s) for i, s in enumerate(mask_select)}
        self.island_c = {o.index: (i, o) for i, o in enumerate(attn_out)}

        oproj_w = re.compile(
            r"encoder_layers_(\d+)_self_attn_o_proj_weight_to_fp16_palettized"
        )
        oproj = {}
        for stmt in self.statements:
            if stmt.op != "linear" or stmt.shape is None:
                continue
            m = oproj_w.fullmatch(
                str(stmt.kwargs.get("weight", "")).strip().strip("'\"")
            )
            if m is None:
                continue
            layer = int(m.group(1))
            if self.island is None or not (
                self.island.bundles / f"island-oproj-L{layer:02d}"
            ).is_dir():
                continue
            oproj[stmt.index] = (layer, stmt)
        self.island_oproj = oproj
        if self.island is not None and "O" in self.placed:
            self.island.resident_bundles.update(
                f"island-oproj-L{layer:02d}" for layer, _ in oproj.values()
            )

    def _index_fusions(self) -> None:
        """Find the conv-module GLU: sigmoid(split_1) consumed by exactly one
        mul whose other operand is the sibling split half. That mul becomes
        one fused dispatch and the sigmoid statement never executes."""
        consumers: dict[str, list[Statement]] = {}
        for stmt in self.statements:
            for token in stmt.kwargs.values():
                for name in self._operand_names(token):
                    consumers.setdefault(name, []).append(stmt)
        for stmt in self.statements:
            if stmt.op != "sigmoid":
                continue
            users = consumers.get(stmt.names[0], [])
            if len(users) != 1 or users[0].op != "mul":
                continue
            mul = users[0]
            b_name = stmt.kwargs["x"].strip()
            a_names = [
                token.strip()
                for key, token in mul.kwargs.items() if key in ("x", "y")
                and token.strip() != stmt.names[0]
            ]
            if len(a_names) != 1:
                continue
            a_name = a_names[0]
            split = self.producer.get(a_name)
            if split is None or split is not self.producer.get(b_name):
                continue
            if split.op != "split":
                continue
            parent = self.producer.get(split.kwargs["x"].strip())
            if parent is None or parent.shape is None or len(parent.shape) != 3:
                continue
            # Only a channel-axis split of a contiguous 3D parent leaves both
            # halves contiguous, which the flat kernel indexing requires.
            axis = self.ints(split.kwargs["axis"])[0]
            if axis % len(parent.shape) != 1 or self.ints(
                split.kwargs["num_splits"]
            )[0] != 2:
                continue
            self.glu_fusions[mul.index] = (a_name, b_name)
            self.glu_sigmoid_done.add(stmt.index)
        # The feed-forward silu: a biased linear whose single consumer is
        # exactly one silu folds the silu into the chain kernel's final
        # store. The linear's fused dispatch writes the silu result under
        # both names; the silu statement becomes a no-op. If the linear
        # cannot take the fused path at apply time it discards the
        # mapping and the silu executes normally.
        if CHAIN_FUSION_ENABLED:
            for stmt in self.statements:
                if stmt.op != "linear" or "bias" not in stmt.kwargs:
                    continue
                users = consumers.get(stmt.names[0], [])
                if len(users) != 1 or users[0].op != "silu":
                    continue
                silu = users[0]
                if silu.kwargs.get("x", "").strip() != stmt.names[0]:
                    continue
                self.linear_silu[stmt.index] = silu.index
                self.silu_done.add(silu.index)

    def _last_use(self) -> None:
        """Index of the final statement that reads each name, so the runner can
        release device memory as it goes. The encoder's constants alone are
        1.2 GB of fp16; holding every intermediate would not fit."""
        self.last_use: dict[str, int] = {}
        self.releases: dict[int, list[str]] = {}
        for stmt in self.statements:
            operands = []
            for token in stmt.kwargs.values():
                for name in self._operand_names(token):
                    operands.append(name)
                    self.last_use[name] = max(
                        self.last_use.get(name, -1), stmt.index
                    )
            stmt.operands = operands
        # The fused mul reads the split sibling after the skipped sigmoid
        # statement, so that operand must live until the mul, not the sigmoid.
        for mul_index, (_a, b_name) in self.glu_fusions.items():
            self.last_use[b_name] = max(
                self.last_use.get(b_name, -1), mul_index
            )
        # Group names by their last-use index so run() retires each name at
        # its own statement instead of rescanning every live name per
        # statement -- identical deletions, O(released) instead of O(all).
        for name, last in self.last_use.items():
            self.releases.setdefault(last, []).append(name)

    def _operand_names(self, token: str) -> list[str]:
        token = token.strip()
        if token.startswith("("):
            return [t.strip() for t in split_top(token.strip("() ")) if t.strip() in self.producer]
        return [token] if token in self.producer else []

    # ------------------------------------------------------------- resolution

    def ensure(self, name: str) -> None:
        """Execute the statement producing ``name`` if it has not run yet.

        Island A's second program needs q_scaled and k_headsT, which the MIL
        schedules after the rel-pos matmul. They are pure functions of values
        already produced, so pulling them forward is semantically identical.
        """
        if name in self.values or name in self.meta:
            return
        stmt = self.producer.get(name)
        if stmt is None:
            raise EncoderRunError(f"unresolved operand {name!r}")
        if stmt.done:
            return
        for operand in stmt.operands:
            self.ensure(operand)
        self.execute(stmt)

    def tensor(self, token: str) -> mx.array:
        token = token.strip()
        self.ensure(token)
        if token in self.values:
            return self.values[token]
        if token in self.meta:
            value = self.meta[token]
            return mx.array(value)
        raise EncoderRunError(f"unresolved tensor {token!r}")

    def scalar(self, token: str):
        """Host-side metadata (shapes, axes, perms, masks, epsilon, flags).

        Only constants are read this way; the pinned encoder derives no shape
        from a computed tensor, so this never forces a device sync on a value
        the run computed.
        """
        token = token.strip()
        self.ensure(token)
        if token in self.meta:
            return self.meta[token]
        raise EncoderRunError(f"{token!r} is not a host-side constant")

    def ints(self, token: str) -> list[int]:
        value = self.scalar(token)
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        return [int(value)]

    def bools(self, token: str) -> list[bool]:
        value = self.scalar(token)
        if isinstance(value, (list, tuple)):
            return [bool(v) for v in value]
        return [bool(value)]

    # ------------------------------------------------------------------ const

    def eval_const(self, stmt: Statement) -> None:
        self._eval_const(stmt)
        if not CONST_CACHE_ENABLED:
            return
        name = stmt.names[0]
        if name in self.values:
            self.const_values[name] = self.values[name]
        if name in self.meta:
            self.const_meta[name] = self.meta[name]

    def _eval_const(self, stmt: Statement) -> None:
        name = stmt.names[0]
        dtype_name, shape = stmt.dtype, stmt.shape
        value_text = stmt.const_kwargs["val"]
        if dtype_name == "string":
            self.meta[name] = re.search(r'"([^"]*)"', value_text).group(1)
            return
        blob = BLOBFILE.search(value_text)
        if blob is None:
            inner = value_text[value_text.index("(") + 1 : value_text.rindex(")")]
            payload = inner.strip("[] ")
            if dtype_name == "bool":
                parsed = [item.strip() == "true" for item in payload.split(",")]
            elif dtype_name == "int32":
                parsed = [int(item) for item in payload.split(",")]
            else:
                parsed = [float(item) for item in payload.split(",")]
            self.meta[name] = parsed if shape else parsed[0]
            self.values[name] = mx.array(
                np.array(parsed, dtype=NP_DTYPES[dtype_name]).reshape(shape)
            )
            return
        count = 1
        for dim in shape:
            count *= dim
        np_dtype = NP_DTYPES[dtype_name]
        raw = self.blobs.read_bytes(
            blob.group("path"), int(blob.group("offset")),
            count * np.dtype(np_dtype).itemsize,
        )
        host = np.frombuffer(raw, dtype=np_dtype, count=count)
        self.values[name] = mx.array(host).reshape(shape) if shape else mx.array(host[0])
        if not shape:
            # Scalar weights are also pad values / multipliers read as metadata.
            self.meta[name] = float(host[0]) if np_dtype != np.int32 else int(host[0])

    # --------------------------------------------------------------- dispatch

    def _pipe(self, *values, op: str = "") -> None:
        if self.pipe and (not self.pipe_ops or op in self.pipe_ops):
            mx.async_eval(*[v for v in values if isinstance(v, mx.array)])

    def execute(self, stmt: Statement) -> None:
        op_started = time.monotonic_ns()
        try:
            self._execute(stmt)
        finally:
            wall = time.monotonic_ns() - op_started
            self.op_wall_ns[stmt.op] = self.op_wall_ns.get(stmt.op, 0) + wall
            if os.environ.get("ANE_STMT_WALL") and wall > 50_000_000:
                print(
                    f"stmt_wall_ms {stmt.index} {stmt.op} {wall / 1e6:.1f} "
                    f"names={stmt.names[:1]}",
                    flush=True,
                )

    def _execute(self, stmt: Statement) -> None:
        if stmt.done:
            return
        stmt.done = True
        if stmt.op == "const":
            self.eval_const(stmt)
            self.executed += 1
            return

        if "A" in self.placed and stmt.index in self.island_a:
            self._run_island_a(stmt)
            return
        if "B" in self.placed and stmt.index in self.island_b:
            self._run_island_b(stmt)
            return
        if "C" in self.placed and stmt.index in self.island_c:
            self._run_island_c(stmt)
            return
        if "O" in self.placed and stmt.index in self.island_oproj:
            self._run_island_oproj(stmt)
            return
        if (
            "A" in self.placed
            and stmt.index in self.island_a_partner
            and stmt.names[0] not in self.values
        ):
            # The content matmul is produced by island A's second program; the
            # splice runs at the rel-pos statement, which comes first.
            self.ensure(self.statements[self.island_a_partner[stmt.index]].names[0])
            if stmt.names[0] in self.values:
                self.executed += 1
                return

        if stmt.op == "sigmoid" and stmt.index in self.glu_sigmoid_done:
            # Consumed only by the fused mul; its value is never read.
            self.executed += 1
            return
        if stmt.op == "silu" and stmt.index in self.silu_done:
            # Produced by the fused chain+bias(+silu) linear dispatch.
            self.executed += 1
            return
        if stmt.index in self.glu_fusions:
            a_name, b_name = self.glu_fusions[stmt.index]
            a = self.tensor(a_name)
            b = self.tensor(b_name)
            n = a.size
            out = _glu_kernel()(
                inputs=[a, b],
                output_shapes=[(n,)],
                output_dtypes=[mx.float16],
                grid=(n, 1, 1),
                threadgroup=(256, 1, 1),
                stream=mx.gpu,
            )[0]
            self.values[stmt.names[0]] = mx.reshape(out, a.shape)
            self._pipe(self.values[stmt.names[0]])
            self.executed += 1
            self.gpu_ops += 1
            return
        if stmt.op == "split":
            parts = self.apply_split(stmt)
            if len(parts) != len(stmt.names):
                raise EncoderRunError(
                    f"split produced {len(parts)} of {len(stmt.names)}"
                )
            self.values.update(zip(stmt.names, parts))
            self._pipe(*parts)
        else:
            self.values[stmt.names[0]] = self.apply(stmt)
            if stmt.index in self.linear_silu:
                self.values[
                    self.statements[self.linear_silu[stmt.index]].names[0]
                ] = self.values[stmt.names[0]]
            self._pipe(self.values[stmt.names[0]], op=stmt.op)
        self.executed += 1
        self.gpu_ops += 1

    def apply_split(self, stmt: Statement) -> list[mx.array]:
        axis = self.ints(stmt.kwargs["axis"])[0]
        count = self.ints(stmt.kwargs["num_splits"])[0]
        return list(mx.split(self.tensor(stmt.kwargs["x"]), count, axis=axis))

    # ----------------------------------------------------------- ANE islands

    def _run_island_a(self, stmt: Statement) -> None:
        layer, scores_stmt, content_stmt = self.island_a[stmt.index]
        if self.bools(scores_stmt.kwargs["transpose_x"])[0] or self.bools(
            scores_stmt.kwargs["transpose_y"]
        )[0]:
            raise EncoderRunError(f"layer {layer} rel-pos matmul is transposed")
        if self.bools(content_stmt.kwargs["transpose_x"])[0] or not self.bools(
            content_stmt.kwargs["transpose_y"]
        )[0]:
            raise EncoderRunError(
                f"layer {layer} content matmul transpose flags are not (x=false, y=true)"
            )
        q_v = self.tensor(scores_stmt.kwargs["x"])
        pos_kT = self.tensor(scores_stmt.kwargs["y"])
        q_scaled = self.tensor(content_stmt.kwargs["x"])
        # The bundle binds the already-transposed key tensor.
        k_headsT = mx.swapaxes(self.tensor(content_stmt.kwargs["y"]), -1, -2)
        results = self.island.submit(
            "island-attn-a-kt", f"L{layer:02d}-A",
            {"q_v": q_v, "pos_kT": pos_kT, "q_scaled": q_scaled, "k_headsT": k_headsT},
            {
                "attention_scores_1": (scores_stmt.shape, "fp16"),
                "matmul_0": (content_stmt.shape, "fp16"),
            },
        )
        self.values[scores_stmt.names[0]] = results["attention_scores_1"]
        self.values[content_stmt.names[0]] = results["matmul_0"]
        content_stmt.done = True
        self.executed += 2
        self.ane_ops += 2

    def _run_island_b(self, stmt: Statement) -> None:
        """The -inf attention-mask select, one submit per layer.

        The MIL writes ``a`` as an fp16 scalar and ``cond`` as [1, 1, 375, 375],
        relying on broadcast. The compiled bundle binds all three operands at
        the full [1, 8, 375, 375], so both are expanded here. That expansion is
        marshalling, not arithmetic: it replicates bytes the run already holds,
        the same way island A hands the bundle an already-transposed key.
        """
        layer, sel_stmt = self.island_b[stmt.index]
        fill = self.tensor(sel_stmt.kwargs["a"])
        cond = self.tensor(sel_stmt.kwargs["cond"])
        matrix_bd = self.tensor(sel_stmt.kwargs["b"])
        if fill.shape != ():
            raise EncoderRunError(f"layer {layer} select fill is {fill.shape}, want scalar")
        if tuple(cond.shape) != (1, 1, 375, 375):
            raise EncoderRunError(f"layer {layer} select cond is {tuple(cond.shape)}")
        if tuple(matrix_bd.shape) != ISLAND_B_SHAPE:
            raise EncoderRunError(f"layer {layer} select b is {tuple(matrix_bd.shape)}")
        if cond.dtype != mx.bool_:
            raise EncoderRunError(f"layer {layer} select cond dtype {cond.dtype}")
        if self.cond_census is None:
            # Measured once, not inherited: whether any lane actually selects
            # the -inf fill decides what this run can claim about the select.
            # var_373 is one tensor shared by all 24 layers, so layer 0 is the
            # whole story, and this reads the host copy rather than adding a
            # device reduction that would perturb the GPU counters.
            host_cond = np.asarray(cond)
            self.cond_census = {
                "cond_tensor": sel_stmt.kwargs["cond"],
                "shape": list(host_cond.shape),
                "elements": int(host_cond.size),
                "true_lanes": int(host_cond.sum()),
                "broadcast_elements": int(host_cond.size) * ISLAND_B_SHAPE[1],
                "broadcast_true_lanes": int(host_cond.sum()) * ISLAND_B_SHAPE[1],
                # repr, not float: -inf is not valid JSON.
                "fill_value": repr(np.asarray(fill).astype(np.float16).item()),
                "fill_bits": "0x%04X" % int(
                    np.asarray(fill).astype(np.float16).view(np.uint16)
                ),
                "shared_across_layers": True,
            }
        results = self.island.submit(
            "island-select-8head", f"L{layer:02d}-B",
            {
                "ninf_rt": mx.contiguous(mx.broadcast_to(fill, ISLAND_B_SHAPE)),
                "matrix_bd_5": matrix_bd,
                "cond": mx.contiguous(mx.broadcast_to(cond, ISLAND_B_SHAPE)),
            },
            {"attention_mask_9": (sel_stmt.shape, "fp16")},
        )
        self.values[sel_stmt.names[0]] = results["attention_mask_9"]
        self.executed += 1
        self.ane_ops += 1

    def _run_island_c(self, stmt: Statement) -> None:
        layer, out_stmt = self.island_c[stmt.index]
        if self.bools(out_stmt.kwargs["transpose_x"])[0] or self.bools(
            out_stmt.kwargs["transpose_y"]
        )[0]:
            raise EncoderRunError(f"layer {layer} PV matmul is transposed")
        results = self.island.submit(
            "island-pv", f"L{layer:02d}-C",
            {
                "probs": self.tensor(out_stmt.kwargs["x"]),
                "v_heads": self.tensor(out_stmt.kwargs["y"]),
            },
            {"attn_output_1": (out_stmt.shape, "fp16")},
        )
        self.values[out_stmt.names[0]] = results["attn_output_1"]
        self.executed += 1
        self.ane_ops += 1

    def _run_island_oproj(self, stmt: Statement) -> None:
        layer, l2 = self.island_oproj[stmt.index]
        x = self.tensor(stmt.kwargs["x"])
        if tuple(x.shape) != (1, 375, 1024):
            raise EncoderRunError(
                f"o-projection input for L{layer:02d}-O is {tuple(x.shape)}"
            )
        results = self.island.submit(
            f"island-oproj-L{layer:02d}", f"L{layer:02d}-O",
            {"x": x}, {"y": (l2.shape, "fp16")},
        )
        self.values[l2.names[0]] = results["y"]
        self.executed += 1
        self.ane_ops += 1

    # ------------------------------------------------------------------- ops

    def apply(self, stmt: Statement) -> mx.array:
        handler = self._OP_HANDLERS.get(stmt.op)
        if handler is None:
            raise EncoderRunError(
                f"unimplemented op {stmt.op!r} ({stmt.dtype} {stmt.shape})"
            )
        return handler(self, stmt)

    def _apply_cast(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        target = self.scalar(kwargs["dtype"])
        if target not in MX_DTYPES:
            raise EncoderRunError(f"cast dtype {target}")
        return tensor(kwargs["x"]).astype(MX_DTYPES[target])

    def _apply_expand_dims(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        out = tensor(kwargs["x"])
        for axis in sorted(self.ints(kwargs["axes"])):
            out = mx.expand_dims(out, axis)
        return out

    def _apply_squeeze(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        return mx.squeeze(tensor(kwargs["x"]), axis=tuple(self.ints(kwargs["axes"])))

    def _apply_reduce(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        axes = tuple(self.ints(kwargs["axes"]))
        keep = self.bools(kwargs["keep_dims"])[0]
        fn = {"reduce_sum": mx.sum, "reduce_min": mx.min, "reduce_max": mx.max}[op]
        return fn(tensor(kwargs["x"]), axis=axes, keepdims=keep)

    def _apply_arith(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        x, y = tensor(kwargs["x"]), tensor(kwargs["y"])
        if x.dtype == mx.bool_ and y.dtype == mx.bool_:
            if op != "mul":
                raise EncoderRunError(f"bool {op}")
            return mx.logical_and(x, y)
        if x.dtype == mx.int32 and y.dtype == mx.int32:
            raw = {"add": x + y, "sub": x - y, "mul": x * y}[op]
            return raw.astype(mx.int32)
        fused = _pw(op, x, y)
        if fused is not None:
            return fused
        fx, fy = x.astype(mx.float32), y.astype(mx.float32)
        return {"add": fx + fy, "sub": fx - fy, "mul": fx * fy}[op].astype(mx.float16)

    def _apply_floor_div(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        x, y = tensor(kwargs["x"]), tensor(kwargs["y"])
        if x.dtype == mx.int32 and y.dtype == mx.int32:
            return mx.floor(x.astype(mx.float32) / y.astype(mx.float32)).astype(mx.int32)
        return mx.floor(
            x.astype(mx.float32) / y.astype(mx.float32)
        ).astype(mx.float16)

    def _apply_floor(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        return mx.floor(tensor(kwargs["x"]).astype(mx.float32)).astype(mx.float16)

    def _apply_less(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        return mx.less(tensor(kwargs["x"]), tensor(kwargs["y"]))

    def _apply_logical_not(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        return mx.logical_not(tensor(kwargs["x"]))

    def _apply_logical_and(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        return mx.logical_and(tensor(kwargs["x"]), tensor(kwargs["y"]))

    def _apply_relu(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        return mx.maximum(tensor(kwargs["x"]).astype(mx.float32), 0.0).astype(mx.float16)

    def _apply_sigmoid(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        x = tensor(kwargs["x"])
        if x.dtype == mx.float16:
            n = x.size
            out = _sigmoid_kernel()(
                inputs=[x],
                output_shapes=[(n,)],
                output_dtypes=[mx.float16],
                grid=(n, 1, 1),
                threadgroup=(256, 1, 1),
                stream=mx.gpu,
            )[0]
            return mx.reshape(out, x.shape)
        return mx.sigmoid(x.astype(mx.float32)).astype(mx.float16)

    def _apply_silu(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        x = tensor(kwargs["x"])
        n = x.size
        out = _silu_kernel()(
            inputs=[x],
            output_shapes=[(n,)],
            output_dtypes=[mx.float16],
            grid=(n, 1, 1),
            threadgroup=(256, 1, 1),
            stream=mx.gpu,
        )[0]
        return mx.reshape(out, x.shape)

    def _apply_transpose(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        x = tensor(kwargs["x"])
        perm = [a % x.ndim for a in self.ints(kwargs["perm"])]
        return mx.contiguous(mx.transpose(x, perm))

    def _apply_reshape(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        return mx.reshape(tensor(kwargs["x"]), self.ints(kwargs["shape"]))

    def _apply_tile(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        return mx.tile(tensor(kwargs["x"]), self.ints(kwargs["reps"]))

    def _apply_concat(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        axis = self.ints(kwargs["axis"])[0]
        if "values" in kwargs:
            names = split_top(kwargs["values"].strip("() "))
        else:
            names = [
                kwargs[key]
                for key in sorted(
                    (k for k in kwargs if re.fullmatch(r"x\d+", k)),
                    key=lambda k: int(k[1:]),
                )
            ]
        return mx.concatenate([tensor(n) for n in names], axis=axis)

    def _apply_linear(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        # fp16 leftover datapath, batched. Default path: one custom
        # coopmat dispatch (_linear_f16_coopmat_kernel) produces every
        # 16-wide block's fp32 partial rounded once to fp16 — the same
        # rounding the f32 batched matmul + fp16 partials chain
        # applied — and _leftover_chain_kernel applies the landed
        # rounding: each block's fp16 partial accumulates in fp16
        # ascending, one dispatch. Byte-identical to the f32-partials
        # batched path (exact fp16->fp32 widening, identical staging
        # values, MMA sequence, and drain); the fp32 matmul route
        # below stays as the guard fallback.
        x = tensor(kwargs["x"])
        weight = tensor(kwargs["weight"])
        k = int(x.shape[-1])
        if k % 16:
            raise EncoderRunError(f"linear K {k} not a multiple of 16")
        blocks = k // 16
        rows = 1
        for dim in x.shape[:-1]:
            rows *= dim
        n_out = int(weight.shape[0])
        if (
            x.dtype == mx.float16
            and weight.dtype == mx.float16
            and 1 <= rows <= 65535 * 32
            and blocks <= 65535
            and n_out <= 65535 * 32
        ):
            lhs = mx.reshape(x, (rows, k))
            rhs = mx.reshape(weight, (n_out, k))
            partials = _linear_f16_coopmat_kernel()(
                inputs=[lhs, rhs],
                output_shapes=[(blocks, rows, n_out)],
                output_dtypes=[mx.float16],
                grid=((n_out + 31) // 32 * 32, (rows + 31) // 32, blocks),
                threadgroup=(32, 1, 1),
                stream=mx.gpu,
            )[0]
            silu_index = (
                self.linear_silu.get(stmt.index) if CHAIN_FUSION_ENABLED else None
            )
            bias_arr = tensor(kwargs["bias"]) if "bias" in kwargs else None
            if (
                CHAIN_FUSION_ENABLED
                and bias_arr is not None
                and bias_arr.dtype == mx.float16
                and n_out % 2 == 0
            ):
                if silu_index is not None:
                    out = _leftover_chain_bias_silu_kernel()(
                        inputs=[partials, bias_arr],
                        output_shapes=[(rows, n_out)],
                        output_dtypes=[mx.float16],
                        grid=(rows * (n_out // 2), 1, 1),
                        threadgroup=(256, 1, 1),
                        stream=mx.gpu,
                    )[0]
                    return mx.reshape(out, tuple(x.shape[:-1]) + (n_out,))
                out = _leftover_chain_bias_kernel()(
                    inputs=[partials, bias_arr],
                    output_shapes=[(rows, n_out)],
                    output_dtypes=[mx.float16],
                    grid=(rows * (n_out // 2), 1, 1),
                    threadgroup=(256, 1, 1),
                    stream=mx.gpu,
                )[0]
                return mx.reshape(out, tuple(x.shape[:-1]) + (n_out,))
            if silu_index is not None:
                # The fused silu will not happen; let the standalone
                # silu statement execute.
                self.silu_done.discard(silu_index)
            out = _leftover_chain_kernel()(
                inputs=[partials],
                output_shapes=[(rows, n_out)],
                output_dtypes=[mx.float16],
                grid=(rows * n_out, 1, 1),
                threadgroup=(256, 1, 1),
                stream=mx.gpu,
            )[0]
            out = mx.reshape(out, tuple(x.shape[:-1]) + (n_out,))
            if "bias" in kwargs:
                out = out.astype(mx.float32) + tensor(kwargs["bias"]).astype(
                    mx.float32
                )
            return out.astype(mx.float16)
        xb = mx.transpose(mx.reshape(x, (rows, blocks, 16)), (1, 0, 2)).astype(
            mx.float32
        )  # [K/16, M, 16]
        wb = mx.transpose(
            mx.reshape(weight, (weight.shape[0], blocks, 16)), (1, 2, 0)
        ).astype(mx.float32)  # [K/16, 16, N]
        partials = xb @ wb  # [K/16, M, N] fp32
        out = _leftover_chain_kernel()(
            inputs=[partials],
            output_shapes=[(rows, weight.shape[0])],
            output_dtypes=[mx.float16],
            grid=(rows * weight.shape[0], 1, 1),
            threadgroup=(256, 1, 1),
            stream=mx.gpu,
        )[0]
        out = mx.reshape(out, tuple(x.shape[:-1]) + (weight.shape[0],))
        silu_index = (
            self.linear_silu.get(stmt.index) if CHAIN_FUSION_ENABLED else None
        )
        bias_arr = tensor(kwargs["bias"]) if "bias" in kwargs else None
        if (
            CHAIN_FUSION_ENABLED
            and bias_arr is not None
            and bias_arr.dtype == mx.float16
            and weight.shape[0] % 2 == 0
        ):
            if silu_index is not None:
                out = _leftover_chain_bias_silu_kernel()(
                    inputs=[partials, bias_arr],
                    output_shapes=[(rows, weight.shape[0])],
                    output_dtypes=[mx.float16],
                    grid=(rows * (weight.shape[0] // 2), 1, 1),
                    threadgroup=(256, 1, 1),
                    stream=mx.gpu,
                )[0]
                out = mx.reshape(out, tuple(x.shape[:-1]) + (weight.shape[0],))
                return out
            out = _leftover_chain_bias_kernel()(
                inputs=[partials, bias_arr],
                output_shapes=[(rows, weight.shape[0])],
                output_dtypes=[mx.float16],
                grid=(rows * (weight.shape[0] // 2), 1, 1),
                threadgroup=(256, 1, 1),
                stream=mx.gpu,
            )[0]
            out = mx.reshape(out, tuple(x.shape[:-1]) + (weight.shape[0],))
            return out
        if silu_index is not None:
            # The fused silu will not happen; let the standalone
            # silu statement execute.
            self.silu_done.discard(silu_index)
        if "bias" in kwargs:
            out = out.astype(mx.float32) + tensor(kwargs["bias"]).astype(mx.float32)
        return out.astype(mx.float16)

    def _apply_matmul(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        a = tensor(kwargs["x"]).astype(mx.float32)
        b = tensor(kwargs["y"]).astype(mx.float32)
        if self.bools(kwargs["transpose_x"])[0]:
            a = mx.swapaxes(a, -1, -2)
        if self.bools(kwargs["transpose_y"])[0]:
            b = mx.swapaxes(b, -1, -2)
        return (a @ b).astype(mx.float16)

    def _apply_layer_norm(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        axes = tuple(self.ints(kwargs["axes"]))
        x = tensor(kwargs["x"])
        if axes != (-1,) or x.ndim != 3 or x.shape[-1] != 1024:
            raise EncoderRunError(
                f"layer_norm form {x.shape} axes {axes} is not the pinned "
                "encoder envelope"
            )
        # Two mx.mean reductions keep the ReduceF32 dispatches and their
        # chunked order bit-identical; the standalone kernels only replace
        # the elementwise chain around them, in the same fp32 ops with the
        # same single fp16 rounding at the end.
        rows = x.size // 1024
        n = x.size
        xf = _ln_cast_kernel()(
            inputs=[x], output_shapes=[(n,)], output_dtypes=[mx.float32],
            grid=(n, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu,
        )[0]
        mean = mx.mean(mx.reshape(xf, (rows, 1024)), axis=-1, keepdims=True)
        t2 = _ln_sq_kernel()(
            inputs=[xf, mean], output_shapes=[(n,)], output_dtypes=[mx.float32],
            grid=(n, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu,
        )[0]
        var = mx.mean(mx.reshape(t2, (rows, 1024)), axis=-1, keepdims=True)
        gamma = tensor(kwargs["gamma"]) if "gamma" in kwargs else None
        beta = tensor(kwargs["beta"]) if "beta" in kwargs else None
        if gamma is None or beta is None:
            raise EncoderRunError("layer_norm without gamma/beta")
        eps_arr = mx.array([float(self.scalar(kwargs["epsilon"]))], dtype=mx.float32)
        y = _ln_tail_kernel()(
            inputs=[xf, mean, var, gamma, beta, eps_arr],
            output_shapes=[(n,)], output_dtypes=[mx.float16],
            grid=(n, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu,
        )[0]
        return mx.reshape(y, x.shape)

    def _apply_softmax(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        axis = self.ints(kwargs["axis"])[0]
        x = tensor(kwargs["x"])
        if axis % x.ndim != x.ndim - 1 or x.ndim != 4 or x.shape[-1] != 375:
            raise EncoderRunError(
                f"softmax form {x.shape} axis {axis} is not the pinned "
                "encoder envelope"
            )
        # The exact ReduceF16 max and the fp32 ReduceF32 sum keep the
        # reduction dispatches; exp and the final divide fuse around them
        # with the same fp32 arithmetic and the same boundary rounding.
        rows = x.size // 375
        n = x.size
        rowmax = mx.max(x, axis=-1, keepdims=True)
        e = _sm_exp_kernel()(
            inputs=[x, rowmax], output_shapes=[(n,)], output_dtypes=[mx.float32],
            grid=(n, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu,
        )[0]
        rowsum = mx.sum(mx.reshape(e, (rows, 375)), axis=-1, keepdims=True)
        y = _sm_div_kernel()(
            inputs=[e, rowsum], output_shapes=[(n,)], output_dtypes=[mx.float16],
            grid=(n, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu,
        )[0]
        return mx.reshape(y, x.shape)

    def _apply_select(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        cond = tensor(kwargs["cond"])
        a = tensor(kwargs["a"]).astype(mx.float32)
        b = tensor(kwargs["b"]).astype(mx.float32)
        return mx.where(cond, a, b).astype(mx.float16)

    def _apply_pad(self, stmt: Statement) -> mx.array:
        op, kwargs = stmt.op, stmt.kwargs
        tensor = self.tensor
        mode = self.scalar(kwargs["mode"])
        if mode != "constant":
            raise EncoderRunError(f"pad mode {mode}")
        pad = self.ints(kwargs["pad"])
        value = float(self.scalar(kwargs["constant_val"]))
        x = tensor(kwargs["x"])
        pairs = [(pad[i], pad[i + 1]) for i in range(0, len(pad), 2)]
        pairs = pairs[-x.ndim :]
        widths = [(0, 0)] * (x.ndim - len(pairs)) + pairs
        return mx.pad(
            x.astype(mx.float32), widths, constant_values=value
        ).astype(mx.float16)

    _OP_HANDLERS = {
        "cast": _apply_cast,
        "expand_dims": _apply_expand_dims,
        "squeeze": _apply_squeeze,
        "reduce_sum": _apply_reduce,
        "reduce_min": _apply_reduce,
        "reduce_max": _apply_reduce,
        "add": _apply_arith,
        "sub": _apply_arith,
        "mul": _apply_arith,
        "floor_div": _apply_floor_div,
        "floor": _apply_floor,
        "less": _apply_less,
        "logical_not": _apply_logical_not,
        "logical_and": _apply_logical_and,
        "relu": _apply_relu,
        "sigmoid": _apply_sigmoid,
        "silu": _apply_silu,
        "transpose": _apply_transpose,
        "reshape": _apply_reshape,
        "tile": _apply_tile,
        "concat": _apply_concat,
        "linear": _apply_linear,
        "matmul": _apply_matmul,
        "conv": lambda self, stmt: self.apply_conv(stmt.kwargs),
        "layer_norm": _apply_layer_norm,
        "softmax": _apply_softmax,
        "select": _apply_select,
        "pad": _apply_pad,
        "slice_by_index": lambda self, stmt: self.apply_slice(stmt.kwargs),
    }

    def apply_conv(self, kwargs: dict) -> mx.array:
        """MIL conv is NCHW; MLX conv is NHWC. Spatial rank 1 lifts to unit H."""
        x = self.tensor(kwargs["x"]).astype(mx.float32)
        weight = self.tensor(kwargs["weight"]).astype(mx.float32)
        bias = self.tensor(kwargs["bias"]).astype(mx.float32) if "bias" in kwargs else None
        strides = self.ints(kwargs["strides"])
        pad = self.ints(kwargs["pad"])
        pad_type = self.scalar(kwargs["pad_type"])
        groups = self.ints(kwargs["groups"])[0]
        dilations = self.ints(kwargs["dilations"])

        rank = x.ndim - 2
        if rank == 1:
            x = mx.expand_dims(x, 2)
            weight = mx.expand_dims(weight, 2)
            strides = [1, strides[0]]
            dilations = [1, dilations[0]]
            pad = [0, 0] + list(pad) if pad_type == "custom" else pad
        elif rank != 2:
            raise EncoderRunError(f"conv spatial rank {rank}")
        if pad_type == "valid":
            pad = [0, 0, 0, 0]
        elif pad_type != "custom":
            raise EncoderRunError(f"conv pad_type {pad_type}")
        if tuple(dilations) != (1, 1):
            raise EncoderRunError(f"conv dilations {dilations}")

        top, bottom, left, right = pad
        xp = mx.pad(x, [(0, 0), (0, 0), (top, bottom), (left, right)], constant_values=0.0)
        # NCHW -> NHWC, (cout, cin/g, kh, kw) -> (cout, kh, kw, cin/g)
        nhwc = mx.transpose(xp, [0, 2, 3, 1])
        ohwi = mx.transpose(weight, [0, 2, 3, 1])
        out = mx.conv2d(
            nhwc, ohwi, stride=(strides[0], strides[1]), padding=0, groups=groups
        )
        out = mx.transpose(out, [0, 3, 1, 2])
        if bias is not None:
            out = out + mx.reshape(bias, (1, -1, 1, 1))
        if rank == 1:
            out = mx.squeeze(out, axis=2)
        return out.astype(mx.float16)

    def apply_slice(self, kwargs: dict) -> mx.array:
        x = self.tensor(kwargs["x"])
        begin = self.ints(kwargs["begin"])
        end = self.ints(kwargs["end"])
        begin_mask = self.bools(kwargs["begin_mask"]) if "begin_mask" in kwargs else None
        end_mask = self.bools(kwargs["end_mask"]) if "end_mask" in kwargs else None
        strides = self.ints(kwargs["strides"]) if "strides" in kwargs else None
        squeeze_mask = (
            self.bools(kwargs["squeeze_mask"]) if "squeeze_mask" in kwargs else None
        )
        index = []
        for axis in range(x.ndim):
            start = None if (begin_mask and begin_mask[axis]) else int(begin[axis])
            stop = None if (end_mask and end_mask[axis]) else int(end[axis])
            step = 1 if strides is None else int(strides[axis])
            if squeeze_mask and squeeze_mask[axis]:
                index.append(int(begin[axis]))
            else:
                index.append(slice(start, stop, step))
        return x[tuple(index)]

    # ------------------------------------------------------------------- run

    # Whole-program surface contract (field-level, from the island-container
    # diff in receipts/2026-09-22-encoder-direct-exec): the stream reads the
    # attention mask from source channel 6 and the dense input features from
    # channel 7, and writes encoder_hidden to destination channel 4 and the
    # output mask to channel 5. The manifest's binding order is the stream's
    # staging order, so positions 0/1 are mask/features in and
    # hidden/mask out.
    WHOLE_INPUTS = ("attention_mask", "input_features")
    WHOLE_OUTPUTS = ("encoder_hidden", "output_mask")
    WHOLE_HIDDEN_SHAPE = (1, 375, 640)
    WHOLE_MASK_SHAPE = (1, 375)

    def _run_whole(self, inputs: dict, wanted: set[str]) -> dict:
        import numpy as np

        started = time.monotonic_ns()
        features = np.asarray(inputs["input_features"], dtype=np.float32)
        mask = np.asarray(inputs["attention_mask"])
        features = np.ascontiguousarray(features.reshape(3000, 128)).astype(
            np.float16
        )
        mask = np.ascontiguousarray(mask.reshape(-1)).astype(np.float16)
        payload = {
            "attention_mask": mask.tobytes(),
            "input_features": features.tobytes(),
        }
        outputs = {
            "encoder_hidden": (self.WHOLE_HIDDEN_SHAPE, "fp16"),
            "output_mask": (self.WHOLE_MASK_SHAPE, "fp16"),
        }
        results = self.island.submit_whole("whole-encoder", payload, outputs)
        self.executed += 1
        hidden = np.asarray(results["encoder_hidden"], dtype=np.float16)
        hidden = hidden.astype(np.float32).reshape(self.WHOLE_HIDDEN_SHAPE)
        keep = {}
        if "encoder_hidden" in wanted:
            keep["encoder_hidden"] = mx.array(hidden)
        if "encoder_mask" in wanted:
            frames = np.asarray(results["output_mask"], dtype=np.float16)
            keep["encoder_mask"] = mx.array(
                (frames > 0.5).astype(np.int32).reshape(self.WHOLE_MASK_SHAPE)
            )
        missing = wanted - set(keep)
        if missing:
            raise EncoderRunError(
                f"whole-encoder submit did not produce {sorted(missing)}"
            )
        return keep

    def run(self, inputs: dict, wanted: set[str], stop_after: str) -> dict:
        if self.whole_manifest is not None:
            return self._run_whole(inputs, wanted)
        if CONST_CACHE_ENABLED and self.const_values:
            # Warm pass: every const is device-resident in the cache; the
            # restore is a dict insert (buffer re-reference), never an
            # upload. Const statements stay done, so eval_const does not
            # re-run; everything else re-executes. The cache also keeps
            # the buffers alive through this pass's release deletions.
            self.values = dict(self.const_values)
            for stmt in self.statements:
                if stmt.op != "const":
                    stmt.done = False
        else:
            if not CONST_CACHE_ENABLED:
                # Knob off: re-run everything each pass, materialization
                # cost included -- the pre-cache behavior per pass.
                for stmt in self.statements:
                    stmt.done = False
            self.values = {}
        for name, value in inputs.items():
            self.values[name] = value
        keep: dict[str, mx.array] = {}
        protected = set(wanted) | {stop_after}
        for stmt in self.statements:
            self.execute(stmt)
            for name in stmt.names:
                if name in wanted and name in self.values:
                    keep[name] = self.values[name]
            # Release anything whose final reader has run: names are grouped
            # by last-use index at parse time, so each statement touches only
            # the names it retires.
            for name in self.releases.get(stmt.index, ()):
                if name not in protected and name in self.values:
                    del self.values[name]
            if stop_after in keep:
                break
        else:
            raise EncoderRunError(f"never reached {stop_after}")
        missing = set(wanted) - keep.keys()
        if missing:
            raise EncoderRunError(f"never produced {sorted(missing)}")
        mx.eval(list(keep.values()))
        if os.environ.get("ANE_OP_WALL"):
            total = sum(self.op_wall_ns.values()) / 1e6
            print(f"op_wall_ms total={total:.0f}", flush=True)
            for op, ns in sorted(self.op_wall_ns.items(), key=lambda kv: -kv[1]):
                print(f"op_wall_ms {op}={ns / 1e6:.1f}", flush=True)
        return keep


def _loaded_libmlx() -> Path | None:
    """Path of the libmlx.so actually mapped into THIS process.

    The dynamic linker, not the import system, picks this file; a stray
    LD_LIBRARY_PATH (or a wheel tree shadowing the venv) silently swaps
    the GPU backend build under a measurement. 2026-09-17: the same
    harness measured a ~2.8x per-statement GPU matmul difference purely
    from which libmlx.so got picked up. Never quote a wall from a run
    whose loaded binary was not verified.
    """
    with open("/proc/self/maps") as fh:
        for line in fh:
            path = line.rstrip("\n").rpartition("  ")[2]
            if path.endswith("/libmlx.so"):
                return Path(path)
    return None


def assert_mlx_binary_identity() -> dict:
    """Record the loaded libmlx identity; fail loudly on mismatch.

    Always returns the resolved {path, sha256, dist_version}; when
    MLX_OMARCHY_EXPECTED_LIBMLX_SHA256 (or _PATH) is set, a mismatch is a
    hard error, never a warning.
    """
    loaded = _loaded_libmlx()
    if loaded is None:
        raise RuntimeError(
            "no libmlx.so is mapped into this process; the encoder runner "
            "cannot attest which GPU backend it is measuring"
        )
    digest = hashlib.sha256(loaded.read_bytes()).hexdigest()
    try:
        dist_version = importlib.metadata.version("mlx-omarchy")
    except importlib.metadata.PackageNotFoundError:
        dist_version = None
    identity = {
        "loaded_libmlx_path": str(loaded),
        "loaded_libmlx_sha256": digest,
        "dist_version": dist_version,
    }
    want_sha = os.environ.get("MLX_OMARCHY_EXPECTED_LIBMLX_SHA256")
    want_path = os.environ.get("MLX_OMARCHY_EXPECTED_LIBMLX_PATH")
    if want_path and os.path.realpath(loaded) != os.path.realpath(want_path):
        raise RuntimeError(
            f"libmlx identity guard: loaded {loaded} but "
            f"MLX_OMARCHY_EXPECTED_LIBMLX_PATH says {want_path}; refusing "
            "to measure on the wrong binary"
        )
    if want_sha and digest != want_sha:
        raise RuntimeError(
            f"libmlx identity guard: loaded {loaded} sha256 {digest} but "
            f"MLX_OMARCHY_EXPECTED_LIBMLX_SHA256 says {want_sha}; refusing "
            "to measure on the wrong binary"
        )
    return identity


def main() -> int:
    mlx_identity = assert_mlx_binary_identity()
    print(
        "libmlx identity: "
        f"{mlx_identity['loaded_libmlx_path']} "
        f"sha256={mlx_identity['loaded_libmlx_sha256'][:16]} "
        f"dist={mlx_identity['dist_version']}",
        flush=True,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--libane", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline-ms", type=int, default=20000)
    parser.add_argument(
        "--no-ane", action="store_true",
        help="Vulkan-only control run: every op stays on the GPU.",
    )
    parser.add_argument(
        "--islands", default="AC",
        help="which islands to place on the ANE, e.g. ABC or AC. Ignored with "
             "--no-ane. AC reproduces the two-island arm from this same script.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="run the encoder this many times in one process (default 1 = "
             "single pass). With MLX_OMARCHY_ENCODER_CONST_CACHE=1, pass 1 "
             "materializes the consts and later passes re-reference the "
             "device-resident cache; with the default (off) every pass "
             "re-materializes.",
    )
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    features = np.load(args.capture / "encoder_input_features.npy")
    mask = np.load(args.capture / "encoder_input_mask.npy")

    island = None
    if not args.no_ane:
        island = AneIsland(
            args.worker, args.libane, args.bundles, args.scratch, args.deadline_ms
        )

    placed = frozenset(args.islands.upper())
    if placed - frozenset("ABC"):
        raise SystemExit(f"--islands {args.islands!r}: unknown island(s)")
    runner = EncoderRunner(
        args.source / "model.mil", args.source / "model-root", island, placed
    )
    inputs = {
        "input_features": mx.array(
            features.astype(np.float32).reshape(1, 3000, 128)
        ),
        "attention_mask": mx.array(mask.astype(np.int32).reshape(1, 3000)),
    }
    passes: list[dict] = []
    started = time.monotonic_ns()
    try:
        for repeat in range(args.repeat):
            before = trace_snapshot()
            mx.reset_peak_memory()
            pass_started = time.monotonic_ns()
            keep = runner.run(
                inputs=inputs,
                wanted={"encoder_hidden", "encoder_mask"},
                stop_after="encoder_mask",
            )
            pass_ns = time.monotonic_ns() - pass_started
            after = trace_snapshot()
            hidden = np.asarray(keep["encoder_hidden"]).astype(np.float32)
            got_mask = np.asarray(keep["encoder_mask"]).astype(np.int32)
            out_dir = args.out / f"pass-{repeat + 1}" if args.repeat > 1 else args.out
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "encoder_hidden.npy", hidden)
            np.save(out_dir / "encoder_mask.npy", got_mask)
            passes.append({
                "pass": repeat + 1,
                "wall_ns": pass_ns,
                "encoder_hidden_sha256": hashlib.sha256(
                    (out_dir / "encoder_hidden.npy").read_bytes()
                ).hexdigest(),
                "gpu_counter_delta": {k: after[k] - before[k] for k in after},
                "active_memory_mb": round(mx.get_active_memory() / 2**20, 1),
                "peak_memory_mb": round(mx.get_peak_memory() / 2**20, 1),
                "rss_mb": round(vm_rss_kb() / 1024, 1),
            })
            print(
                f"pass {repeat + 1}/{args.repeat}: {pass_ns / 1e6:.1f} ms "
                f"hidden={passes[-1]['encoder_hidden_sha256'][:16]} "
                f"active={passes[-1]['active_memory_mb']}MB "
                f"peak={passes[-1]['peak_memory_mb']}MB "
                f"rss={passes[-1]['rss_mb']}MB",
                flush=True,
            )
    finally:
        if island is not None:
            island.close()
    wall_ns = passes[-1]["wall_ns"]
    gpu_delta = passes[-1]["gpu_counter_delta"]
    report = {
        "layers": runner.layers,
        "ops_executed": runner.executed,
        "gpu_ops": runner.gpu_ops,
        "ane_ops": runner.ane_ops,
        "cpu_tensor_events": runner.cpu_tensor_events,
        "wall_ns": wall_ns,
        "ane_mode": not args.no_ane,
        "islands_placed": "".join(sorted(runner.placed)),
        "gpu_counters": gpu_delta,
        "island_b_cond_census": runner.cond_census,
        "encoder_hidden_shape": list(hidden.shape),
        "repeat": args.repeat,
        "passes": passes,
        "total_wall_ns": time.monotonic_ns() - started,
    }
    if island is not None:
        report["ane"] = {
            "mode": island._mode,
            "submissions": island.submissions,
            "rounds": island.rounds,
            "worker_starts": island.worker_starts,
            "timeouts": island.timeouts,
            "batch_open_ns": island.batch_open_ns,
            "marshal_ns": island.marshal_ns,
            "back_ns": island.back_ns,
            "op_wall_ms": {
                op: round(ns / 1e6, 1)
                for op, ns in sorted(
                    runner.op_wall_ns.items(), key=lambda kv: -kv[1]
                )
            },
            "input_bytes": island.input_bytes,
            "output_bytes": island.output_bytes,
            "exec_ns": island.exec_ns,
            "log": island.log,
        }
    (args.out / "run-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "ane"}, indent=2))
    if island is not None:
        print(
            f"ane mode={island._mode} submissions={island.submissions} "
            f"rounds={island.rounds} worker_starts={island.worker_starts} "
            f"timeouts={island.timeouts} exec_ms={island.exec_ns / 1e6:.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
