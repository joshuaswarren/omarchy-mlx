#!/usr/bin/env python3
"""Store BatchKVCache.offset as a host-built array (oMLX decode host-join fix, opt-in).

Root cause (receipts/2026-10-07-batch-decode-2, term (a)): BatchKVCache keeps
offset as a lazy device array (`self.offset + keys.shape[2]` every step). The
omarchy RoPE trig gate must bound |offset| on the host before it dispatches,
and an offset with a primitive behind it can only be read after a
stream-ordered synchronize: one host join per RoPE call inside the decode
graph. mlx-lm's generate_step passes an int offset and pays none; oMLX serves
every request (c1 too) through BatchGenerator and BatchKVCache.

The values are known on the host: offset_list mirrors offset after every
in-class update (init, update_and_fetch, prepare, finalize, trim, filter,
extend, merge). With MLX_OMARCHY_KV_HOST_OFFSET=1 those sites store
mx.array(offset_list) - the same int32 values with no primitive, which the
gate reads directly. offset becomes a property whose setter clears the mirror,
so any assignment from outside the class (oMLX copies, state restores) keeps
the assigned array and the old behavior.

Default off (read once at import). Apply after patch-mlx-lm-kv-maskless.py.
Usage: python3 patch-mlx-lm-kv-host-offset.py /path/to/venv
"""
import ast
import glob
import sys

MARKER = "MLX_OMARCHY_KV_HOST_OFFSET"

FLAG_OLD = '_KV_MASKLESS_UNPADDED = os.environ.get("MLX_OMARCHY_KV_MASKLESS", "1") != "0"\n'
FLAG_NEW = FLAG_OLD + '_KV_HOST_OFFSET = os.environ.get("MLX_OMARCHY_KV_HOST_OFFSET", "0") == "1"\n'

PROP_OLD = """    @left_padding.setter
    def left_padding(self, value):
        self._left_padding = value
        self.left_padding_list = None
"""
PROP_NEW = PROP_OLD + """
    # offset_list mirrors offset on the host. With MLX_OMARCHY_KV_HOST_OFFSET=1
    # in-class updates store mx.array(offset_list), which the RoPE gate reads
    # without a host join. An outside assignment clears the mirror.
    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value
        self.offset_list = None

    def _set_offset(self, device, host):
        self._offset = mx.array(host) if host is not None and _KV_HOST_OFFSET else device
        self.offset_list = host

    def _shift_offset(self, device, delta):
        # delta: an int for every row, a per-row list, or None (unknown on the host).
        m = self.offset_list
        if m is not None and delta is not None:
            m = [o + d for o, d in zip(m, [delta] * len(m) if isinstance(delta, int) else delta)]
        self._set_offset(device, None if delta is None else m)
"""

SITES = [
    ("""        self.offset = mx.array([-l for l in left_padding])
""", """        lp = self.left_padding_list
        self._set_offset(mx.array([-l for l in left_padding]), None if lp is None else [-l for l in lp])
"""),
    ("""        self.offset = self.offset + keys.shape[2]
        self._idx += keys.shape[2]
""", """        self._shift_offset(self.offset + keys.shape[2], keys.shape[2])
        self._idx += keys.shape[2]
"""),
    ("""            self.offset = self.offset - left_padding
""", """            self._shift_offset(self.offset - left_padding, None if added is None else [-a for a in added])
"""),
    ("""            self.offset = self.offset - padding
""", """            right = self._right_padding_list
            self._shift_offset(self.offset - padding, None if right is None else [-r for r in right])
"""),
    ("""        self.offset = self.offset - n
""", """        self._shift_offset(self.offset - n, -n)
"""),
    ("""        self.offset = self.offset[batch_indices]
""", """        rows, mirror = _host_list(batch_indices), self.offset_list
        self._set_offset(
            self.offset[batch_indices], None if rows is None or mirror is None else [mirror[i] for i in rows]
        )
"""),
    ("""            self.offset = mx.concatenate([self.offset, other.offset])
""", """            offs = (self.offset_list, other.offset_list)
            self._set_offset(
                mx.concatenate([self.offset, other.offset]), None if None in offs else offs[0] + offs[1]
            )
"""),
    ("""        mirror = None
        if self.left_padding_list is not None and other.left_padding_list is not None:
""", """        offs = (self.offset_list, other.offset_list)
        mirror = None
        if self.left_padding_list is not None and other.left_padding_list is not None:
"""),
    ("""        self.left_padding_list = mirror
        self._idx = max_idx
""", """        self.left_padding_list = mirror
        self._set_offset(self._offset, None if None in offs else offs[0] + offs[1])
        self._idx = max_idx
"""),
    ("""        cache.offset += keys.shape[2]
""", """        cache._shift_offset(cache.offset + keys.shape[2], keys.shape[2])
"""),
]

venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
if not site:
    sys.exit("mlx_lm/models not found under " + venv)
q = site[0] + "/cache.py"
text = open(q).read()
if MARKER in text:
    print("already patched:", q)
    sys.exit(0)
if "MLX_OMARCHY_KV_MASKLESS" not in text:
    sys.exit("cache.py is not kv-maskless patched (apply patch-mlx-lm-kv-maskless.py first)")
if text.count(FLAG_OLD) != 1:
    sys.exit("unrecognized kv-maskless flag line in " + q)
text = text.replace(FLAG_OLD, FLAG_NEW, 1)

start = text.find("class BatchKVCache(_BaseCache):\n")
end = text.find("\nclass ", start + 1)
if start < 0 or end < 0:
    sys.exit("BatchKVCache not found in " + q)
body = text[start:end]
for old, new in [(PROP_OLD, PROP_NEW)] + SITES:
    if body.count(old) != 1:
        sys.exit("unrecognized BatchKVCache content in " + q + "; refusing to patch (mlx-lm version mismatch?)")
    body = body.replace(old, new, 1)
text = text[:start] + body + text[end:]
ast.parse(text)
open(q, "w").write(text)
print("patched:", q)
