# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Exactness checks for the LSTM thread rebalance in the GPU TDT loop.

The loop kernel's LSTM chains previously ran only on threads 0-639 with
all four gate folds inline per lane thread; they now map all 2560
``(lane, gate)`` fold items across all 1024 threads (items ``p = t``,
``t + 1024``, ``t + 2048``) and stage each gate pre-activation through a
bit-exact carrier before the unchanged gate-pairing phase reads it back:

* gate 0 -> ``s_relu[lane]``          (native fp16 slot, dead in chains)
* gate 1 -> ``s_bval[lane]``          (float carrier of an fp16 value)
* gate 2 -> ``s_bidx[lane]``          (uint carrier via floatBitsToUint)
* gate 3 -> ``s_h1[lane]``            (float carrier; pairing reads it
  back before overwriting the slot with the layer's h1, same thread)

The fold, bias add, gate pairing, cell update, argmax and control are
untouched, so per ``(lane, gate)`` the arithmetic sequence is identical;
only thread mapping and carrier round-trips differ.  These tests pin
exactly-onto coverage of the item mapping and bit-exactness of every
carrier.
"""

import re
import unittest

import numpy as np

_LANES = 640
_GATES = 4
_ITEMS = _LANES * _GATES
_THREADS = 1024

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.vulkan_tdt_loop import LOOP_WORKGROUP_MEMORY_BYTES, _loop_glsl


def _thread_items(t):
    """Items the rebalanced kernel assigns to thread ``t``."""
    return [p for p in (t, t + _THREADS, t + 2 * _THREADS) if p < _ITEMS]


_TYPE_BYTES = {"float": 4, "float16_t": 2, "uint": 4, "int": 4}


def _rendered_shader():
    """The real shader source the fork compiles (no GPU needed to render)."""
    return _loop_glsl()


def _declared_arrays(src):
    """Parse actual threadgroup declarations: name -> (type, elements)."""
    pairs = re.findall(
        r"threadgroup\s+(\S+)\s+(\w+)\[(\d+)\];", src)
    return {name: (typ, int(n)) for typ, name, n in pairs}


def _guard_holds(src):
    """Code-linked checks: (ok, reason) against the REAL rendered shader."""
    src = re.sub(r"\s+", " ", src)
    if "for (uint p = t; p < 2560u; p += 1024u)" not in src:
        return False, "thread mapping loop bound/stride changed"
    chains = src.find("for (uint p = t; p < 2560u")
    pairing = src.find("for (uint lane = t; lane < 640u; lane += 1024u)", chains)
    if chains < 0 or pairing < 0:
        return False, "chains or pairing phase missing"
    barrier = src.find("threadgroup_barrier(mem_flags::mem_threadgroup);", chains)
    if not (chains < barrier < pairing):
        return False, "staging barrier between chains and pairing missing"
    writes = (
        "s_relu[lane] = pr;", "s_bval[lane] = float(pr);",
        "s_bidx[lane] = floatBitsToUint(float(pr));", "s_h1[lane] = float(pr);",
    )
    for w in writes:
        if w not in src[chains:pairing]:
            return False, f"carrier write missing from chains phase: {w}"
    reads = (
        "float16_t pr0 = s_relu[lane];",
        "float16_t pr1 = float16_t(s_bval[lane]);",
        "float16_t pr2 = float16_t( uintBitsToFloat(s_bidx[lane]));",
        "float16_t pr3 = float16_t(s_h1[lane]);",
    )
    for r in reads:
        if r not in src[pairing:]:
            return False, f"carrier read missing from pairing phase: {r}"
    h1_read = src.find("float16_t pr3 = float16_t(s_h1[lane]);", pairing)
    h1_write = src.find("s_h1[lane] = float(h1);", pairing)
    if not (0 <= h1_read < h1_write):
        return False, "s_h1 carrier read must precede the h1 store"
    return True, "mapping, staging barrier, carriers, and h1 order all present"


class RenderedShaderGuard(unittest.TestCase):
    """Code-linked guard over the REAL rendered kernel (no GPU required).

    These tests import the module and render ``_loop_glsl()`` — the exact
    source the fork compiles — so reverting the kernel, breaking the
    thread mapping, dropping the staging barrier, or changing a carrier
    fails here instead of only on device.
    """

    def test_rendered_shader_carries_rebalance(self):
        ok, reason = _guard_holds(_rendered_shader())
        self.assertTrue(ok, reason)

    def test_guard_is_load_bearing(self):
        """Failing-first: each targeted mutation breaks the guard."""
        src = _rendered_shader()
        chains = src.find("for (uint p = t; p < 2560u")
        pairing = src.find("for (uint lane = t; lane < 640u; lane += 1024u)", chains)
        barrier = src.find(
            "threadgroup_barrier(mem_flags::mem_threadgroup);", chains)
        mutations = {
            "staging barrier removed":
                src[:barrier] + src[barrier + len(
                    "threadgroup_barrier(mem_flags::mem_threadgroup);"):],
            "mapping bound shrunk": src.replace("p < 2560u", "p < 2048u", 1),
            "gate 2 carrier write dropped":
                src.replace("s_bidx[lane] = floatBitsToUint(float(pr));", "", 1),
        }
        for name, mutated in mutations.items():
            with self.subTest(mutation=name):
                ok, reason = _guard_holds(mutated)
                self.assertFalse(ok, f"guard failed to detect: {name}: {reason}")

    def test_budget_derived_from_declarations(self):
        """Byte sum comes from the shader's own declarations, not constants."""
        arrays = _declared_arrays(_rendered_shader())
        self.assertEqual(
            set(arrays),
            {"s_hidden", "s_cell", "s_h1", "s_a", "s_pj", "s_relu",
             "s_bval", "s_bidx", "s_dval", "s_ctl"},
        )
        total = sum(n * _TYPE_BYTES[typ] for typ, n in arrays.values())
        self.assertLessEqual(total, LOOP_WORKGROUP_MEMORY_BYTES)
        self.assertLessEqual(LOOP_WORKGROUP_MEMORY_BYTES, 32768)


class LstmRebalanceTest(unittest.TestCase):
    def test_item_mapping_covers_every_lane_gate_exactly_once(self):
        """The 1024-thread mapping covers all (lane, gate) items exactly once."""
        seen = {}
        for t in range(_THREADS):
            for p in _thread_items(t):
                gate, lane = divmod(p, _LANES)
                self.assertNotIn((lane, gate), seen, f"item {p} duplicated")
                seen[(lane, gate)] = t
        self.assertEqual(len(seen), _ITEMS)
        for lane in range(_LANES):
            for gate in range(_GATES):
                self.assertIn((lane, gate), seen)
        # balance: 512 threads carry three items, 512 carry two; no thread
        # carries more than ceil(2560/1024) = 3 of the 4-gate work
        loads = [len(_thread_items(t)) for t in range(_THREADS)]
        self.assertEqual(loads.count(3), 512)
        self.assertEqual(loads.count(2), 512)
        self.assertEqual(max(loads), 3)

    def test_carriers_roundtrip_fp16_bits(self):
        """Native, float-carried, and uint-carried slots preserve fp16 bits."""
        rng = np.random.default_rng(925)
        values = rng.standard_normal(_LANES).astype(np.float16)

        native = np.array(values, dtype=np.float16)  # s_relu[lane]
        np.testing.assert_array_equal(
            native.view(np.uint16), values.view(np.uint16)
        )

        float_carried = values.astype(np.float32).astype(np.float16)
        np.testing.assert_array_equal(
            float_carried.view(np.uint16), values.view(np.uint16)
        )

        # s_bidx[lane] = floatBitsToUint(float(pr)); read back via
        # uintBitsToFloat then float16_t() — bit-transparent round trip.
        as_uint = values.astype(np.float32).view(np.uint32).copy()
        back = as_uint.view(np.float32).astype(np.float16)
        np.testing.assert_array_equal(back.view(np.uint16), values.view(np.uint16))

        # bias / sign / subnormal edges must survive the float carriers too
        edges = np.array(
            [0.0, -0.0, 1.0, -1.0, 6.0e-8, -6.0e-8, 65504.0, -65504.0,
             5.9604645e-8, -5.9604645e-8, 0.33325195],
            dtype=np.float16,
        )
        for v in edges:
            self.assertEqual(
                np.float16(np.float32(v)).view(np.uint16), v.view(np.uint16)
            )
            self.assertEqual(
                np.float32(v).view(np.uint32).view(np.float32).astype(np.float16).view(np.uint16),
                v.view(np.uint16),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
