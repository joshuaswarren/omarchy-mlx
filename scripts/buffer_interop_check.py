#!/usr/bin/env python3
"""Packed Python buffer consumer boundary checks.

Narrows the buffer-interop contract (previously swept broadly in the
kv-state slice-views work) to the export boundary in
patches/mlx-python-buffer.patch:

1. PyBUF_SIMPLE on a C-contiguous array: zero-copy values, readonly.
2. PyBUF_SIMPLE / PyBUF_ND on strided logical arrays (transpose,
   broadcast, inner slice): owned C-contiguous copy with correct C-order
   values, read-only (detached from the array), held until
   release_buffer (valid read after the source reference is dropped).
   PyBUF_ND also exports shape.
3. PyBUF_STRIDES export stays zero-copy strided and writable:
   memoryview and raw stride requests see the logical strided values.
4. Explicit C/F/ANY_CONTIGUOUS requests: correct layout when the array
   has it, BufferError otherwise (PEP 3118). PyBUF_WRITABLE succeeds on
   dense and strides-aware exports, is refused with BufferError when
   packing would be required.

Flag-exact requests go through a small C shim compiled against the
interpreter's own Python.h (ctypes.pythonapi misbinds PyObject_GetBuffer
on this box's patched CPython, corrupting the view struct; all struct
reads go back through the shim so only its layout matters). Valid
requests only: acquire/read/release, no writes through read-only views.
Usage: python3 scripts/buffer_interop_check.py ; exit 0 = all pass.
"""
import ctypes
import gc
import shlex
import subprocess
import sys
import sysconfig
import tempfile

import numpy as np
import mlx.core as mx

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} {detail}")
        FAILURES.append(name)


SHIM_C = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

size_t shim_pybuffer_size(void) { return sizeof(Py_buffer); }

int shim_getbuf(PyObject *obj, void *v, int flags) {
  return PyObject_GetBuffer(obj, (Py_buffer *)v, flags);
}

void shim_release(void *v) { PyBuffer_Release((Py_buffer *)v); }

Py_ssize_t shim_len(void *v) { return ((Py_buffer *)v)->len; }
Py_ssize_t shim_itemsize(void *v) { return ((Py_buffer *)v)->itemsize; }
int shim_readonly(void *v) { return ((Py_buffer *)v)->readonly; }
int shim_ndim(void *v) { return ((Py_buffer *)v)->ndim; }
int shim_has_shape(void *v) { return ((Py_buffer *)v)->shape != NULL; }
int shim_has_strides(void *v) { return ((Py_buffer *)v)->strides != NULL; }
Py_ssize_t shim_shape(void *v, int i) { return ((Py_buffer *)v)->shape[i]; }
Py_ssize_t shim_stride(void *v, int i) { return ((Py_buffer *)v)->strides[i]; }
const char *shim_format(void *v) { return ((Py_buffer *)v)->format; }
void shim_read(void *v, void *out, Py_ssize_t n) {
  memcpy(out, ((Py_buffer *)v)->buf, (size_t)n);
}
"""

PyBUF_SIMPLE = 0
PyBUF_WRITABLE = 0x0001
PyBUF_FORMAT = 0x0004
PyBUF_ND = 0x0008
PyBUF_STRIDES = 0x0010 | PyBUF_ND
PyBUF_C_CONTIGUOUS = 0x0020 | PyBUF_STRIDES
PyBUF_F_CONTIGUOUS = 0x0040 | PyBUF_STRIDES
PyBUF_ANY_CONTIGUOUS = 0x0080 | PyBUF_STRIDES


def build_shim():
    paths = sysconfig.get_paths()
    cc = shlex.split(sysconfig.get_config_var("CC") or "cc")
    with tempfile.TemporaryDirectory(prefix="bufshim-") as tmp:
        src = f"{tmp}/shim.c"
        so = f"{tmp}/shim.so"
        with open(src, "w") as f:
            f.write(SHIM_C)
        subprocess.run(
            cc + ["-shared", "-fPIC", "-I", paths["include"], src, "-o", so],
            check=True,
            capture_output=True,
        )
        # PyDLL keeps the GIL held across calls; mlx's getbuffer performs
        # its own gil_scoped_release and requires the GIL on entry.
        lib = ctypes.PyDLL(so)
    lib.shim_pybuffer_size.restype = ctypes.c_size_t
    lib.shim_getbuf.argtypes = [ctypes.py_object, ctypes.c_void_p, ctypes.c_int]
    lib.shim_getbuf.restype = ctypes.c_int
    for name in ("shim_len", "shim_itemsize"):
        fn = getattr(lib, name)
        fn.restype = ctypes.c_ssize_t
        fn.argtypes = [ctypes.c_void_p]
    for name in ("shim_readonly", "shim_ndim", "shim_has_shape",
                 "shim_has_strides"):
        fn = getattr(lib, name)
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p]
    for name in ("shim_shape", "shim_stride"):
        fn = getattr(lib, name)
        fn.restype = ctypes.c_ssize_t
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.shim_format.restype = ctypes.c_char_p
    lib.shim_format.argtypes = [ctypes.c_void_p]
    lib.shim_read.restype = None
    lib.shim_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ssize_t]
    lib.shim_release.restype = None
    lib.shim_release.argtypes = [ctypes.c_void_p]
    return lib


lib = build_shim()
VIEW_SIZE = lib.shim_pybuffer_size()


class View:
    """Opaque Py_buffer holder sized by the shim; every field read goes
    back through the shim."""

    def __init__(self):
        self._buf = ctypes.create_string_buffer(VIEW_SIZE)
        self.len = 0

    def read(self):
        out = ctypes.create_string_buffer(self.len)
        lib.shim_read(self._buf, out, self.len)
        return out.raw[:self.len]

    @property
    def readonly(self):
        return bool(lib.shim_readonly(self._buf))

    @property
    def itemsize(self):
        return lib.shim_itemsize(self._buf)

    @property
    def ndim(self):
        return lib.shim_ndim(self._buf)

    @property
    def has_shape(self):
        return bool(lib.shim_has_shape(self._buf))

    @property
    def has_strides(self):
        return bool(lib.shim_has_strides(self._buf))

    @property
    def shape(self):
        return [lib.shim_shape(self._buf, i) for i in range(self.ndim)]

    @property
    def strides(self):
        return [lib.shim_stride(self._buf, i) for i in range(self.ndim)]

    @property
    def format(self):
        return lib.shim_format(self._buf)


def acquire(obj, flags):
    """Return (rc, view); on rc == -1 the second element is the exporter's
    exception type name (PyDLL re-raises it here)."""
    view = View()
    try:
        rc = lib.shim_getbuf(ctypes.py_object(obj), view._buf, flags)
    except BaseException as e:
        return -1, type(e).__name__
    view.len = lib.shim_len(view._buf)
    return rc, view


def release(view):
    lib.shim_release(view._buf)


def c_bytes(arr):
    return np.ascontiguousarray(np.asarray(arr)).tobytes()


base = mx.arange(24, dtype=mx.float32).reshape(4, 6)
mx.eval(base)
base_np = np.asarray(base)
itemsize = base_np.itemsize

print("== 1. SIMPLE on C-contiguous ==")
rc, view = acquire(base, PyBUF_SIMPLE)
check("SIMPLE contig succeeds", rc == 0, str(view if rc else ""))
if rc == 0:
    check("SIMPLE contig values", view.read() == base_np.tobytes())
    check("SIMPLE contig len", view.len == base_np.nbytes)
    check("SIMPLE contig itemsize holds original format value",
          view.itemsize == itemsize)
    check("SIMPLE contig no format", view.format is None)
    check("SIMPLE contig no strides", not view.has_strides)
    check("SIMPLE contig zero-copy is writable", not view.readonly)
    release(view)

for name, arr, packed_expected in [
    ("transpose", base.transpose(), True),
    ("broadcast", mx.broadcast_to(base[0:1], (4, 6)), True),
    # A gapped slice may eval to a row-contiguous array on backends that
    # densify; readonly then correctly stays false (no packing happened).
    ("inner slice", base[:, 1:4], None),
]:
    mx.eval(arr)
    rc, view = acquire(arr, PyBUF_SIMPLE)
    ok = rc == 0
    check(f"SIMPLE {name} succeeds", ok, str(view if not ok else ""))
    if ok:
        check(f"SIMPLE {name} packed C-order values",
              view.read() == c_bytes(arr))
        if packed_expected is not None:
            check(f"SIMPLE {name} packed copy read-only",
                  view.readonly == packed_expected)
        release(view)

lazy = base.transpose()
rc, view = acquire(lazy, PyBUF_SIMPLE)
ok = rc == 0
check("SIMPLE lazy transpose succeeds (export evaluates)", ok,
      str(view if not ok else ""))
if ok:
    check("SIMPLE lazy transpose values", view.read() == c_bytes(lazy))
    release(view)

rc, view = acquire(base.transpose(), PyBUF_ND)
ok = rc == 0
check("ND strided succeeds", ok, str(view if not ok else ""))
if ok:
    check("ND strides stay NULL", not view.has_strides)
    check("ND shape exported", view.has_shape)
    check("ND shape values", view.shape == [6, 4])
    check("ND packed C-order values",
          view.read() == c_bytes(base.transpose()))
    check("ND packed copy read-only", view.readonly)
    release(view)

print("== 3. copy lifetime held until release_buffer ==")
t = base.transpose()
rc, view = acquire(t, PyBUF_SIMPLE)
ok = rc == 0
check("SIMPLE strided acquire for lifetime", ok, str(view if not ok else ""))
if ok:
    expected = view.read()
    del t, lazy
    gc.collect()
    check("owned copy readable after source refs dropped",
          view.read() == expected)
    release(view)

row_slice = base[1:3]
mx.eval(row_slice)
rc, view = acquire(row_slice, PyBUF_SIMPLE)
check("SIMPLE row-contiguous slice succeeds", rc == 0)
if rc == 0:
    check("SIMPLE row-contiguous slice values",
          view.read() == c_bytes(row_slice))
    release(view)

print("== 4. STRIDES stays zero-copy strided ==")
t = base.transpose()
expected_strides = [1 * itemsize, 6 * itemsize]
rc, view = acquire(t, PyBUF_STRIDES | PyBUF_FORMAT)
ok = rc == 0
check("STRIDES strided succeeds", ok, str(view if not ok else ""))
if ok:
    check("STRIDES exported", view.has_strides)
    check("STRIDES values", view.strides == expected_strides)
    check("STRIDES shape exported", view.has_shape)
    check("STRIDES format", view.format == b"f")
    check("STRIDES zero-copy is writable", not view.readonly)
    release(view)
mv = memoryview(t)
check("memoryview strided values", np.asarray(mv).tobytes() == c_bytes(t))
check("np.asarray strides path", np.asarray(t).tobytes() == c_bytes(t))
check("bytes() simple path", bytes(t) == c_bytes(t))

print("== 5. explicit contiguity and writability flags ==")
rc, view = acquire(base, PyBUF_C_CONTIGUOUS)
check("C_CONTIGUOUS on C-contig succeeds", rc == 0)
if rc == 0:
    release(view)
rc, exc = acquire(base, PyBUF_F_CONTIGUOUS)
check("F_CONTIGUOUS on C-contig 2-D refused",
      rc == -1 and exc == "BufferError", str(exc))
t = base.transpose()
rc, view = acquire(t, PyBUF_F_CONTIGUOUS)
check("F_CONTIGUOUS on F-contig transpose succeeds", rc == 0)
if rc == 0:
    release(view)
rc, exc = acquire(t, PyBUF_C_CONTIGUOUS)
check("C_CONTIGUOUS on strided refused", rc == -1 and exc == "BufferError",
      str(exc))
rc, exc = acquire(mx.broadcast_to(base[0:1], (4, 6)), PyBUF_ANY_CONTIGUOUS)
check("ANY_CONTIGUOUS on broadcast refused",
      rc == -1 and exc == "BufferError", str(exc))
vec = mx.arange(6, dtype=mx.float32)
for flag, label in [(PyBUF_C_CONTIGUOUS, "C"), (PyBUF_F_CONTIGUOUS, "F"),
                    (PyBUF_ANY_CONTIGUOUS, "ANY")]:
    rc, view = acquire(vec, flag)
    check(f"{label}_CONTIGUOUS on 1-D succeeds", rc == 0)
    if rc == 0:
        release(view)

rc, view = acquire(base, PyBUF_WRITABLE | PyBUF_SIMPLE)
check("WRITABLE dense SIMPLE succeeds", rc == 0)
if rc == 0:
    check("WRITABLE dense SIMPLE writable", not view.readonly)
    release(view)
rc, exc = acquire(base.transpose(), PyBUF_WRITABLE | PyBUF_SIMPLE)
check("WRITABLE strided SIMPLE refused",
      rc == -1 and exc == "BufferError", str(exc))
rc, view = acquire(base.transpose(), PyBUF_WRITABLE | PyBUF_STRIDES)
check("WRITABLE strided STRIDES retained", rc == 0)
if rc == 0:
    check("WRITABLE strided STRIDES writable", not view.readonly)
    release(view)

if FAILURES:
    print(f"{len(FAILURES)} FAILURES")
    sys.exit(1)
print("ALL CHECKS PASS")
