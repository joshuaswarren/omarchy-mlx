# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Process-wide backend trace counters: one ctypes mirror of the C ABI.

Single shared mirror of ``MlxOmarchyTraceSnapshot``
(``mlx/backend/omarchy/trace.h``) for every python reader. The C writer
fills the struct field-by-field, so a mirror smaller than the library's
struct is a silent heap overflow of 16 bytes per appended field — the
2026-10-04 wave-scheduler GC double-track crash, where the shipped
mirrors stayed at 8 fields while libmlx grew to 10 and the packaged
parakeet worker died at its first GC pass. The loader therefore resolves
``mlx_omarchy_trace_snapshot_abi_size`` from libmlx and refuses to read
on any size mismatch: ABI drift becomes a named error, never corruption.
"""

import ctypes
import importlib.metadata
import os
from functools import cache

_FIELDS = (
    "gpu_primitive_dispatches",
    "vk_submissions",
    "vk_buffer_copies",
    "vk_buffer_fills",
    "vk_compute_dispatches",
    "omarchy_finalize_calls",
    "commit_calls_with_work",
    "commit_calls_noop",
    "barriers_emitted",
    "barriers_skipped",
)


class TraceSnapshot(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in _FIELDS]


def library_path() -> str:
    override = os.environ.get("MLX_OMARCHY_LIBMLX")
    if override:
        return override
    distribution = importlib.metadata.distribution("mlx-omarchy")
    return str(distribution.locate_file("mlx/lib/libmlx.so"))


@cache
def _trace_function():
    library = ctypes.CDLL(library_path())
    size_probe = library.mlx_omarchy_trace_snapshot_abi_size
    size_probe.argtypes = []
    size_probe.restype = ctypes.c_uint64
    library_size = int(size_probe())
    mirror_size = ctypes.sizeof(TraceSnapshot)
    if library_size != mirror_size:
        raise RuntimeError(
            f"MlxOmarchyTraceSnapshot ABI drift: {library_path()} writes "
            f"{library_size} bytes, the python mirror is {mirror_size} bytes; "
            "refusing to read (a stale mirror would corrupt the heap)"
        )
    function = library.mlx_omarchy_trace_snapshot
    function.argtypes = [ctypes.POINTER(TraceSnapshot)]
    function.restype = None
    return function


def trace_snapshot() -> dict[str, int]:
    snapshot = TraceSnapshot()
    _trace_function()(ctypes.byref(snapshot))
    return {name: int(getattr(snapshot, name)) for name in _FIELDS}
